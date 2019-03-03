import numpy as np
import torch
from torch import nn as nn

from bab_runner import device
from plnn.mini_net import Net


class VerificationNetwork(nn.Module):
    def __init__(self, base_network, batch_size, input_size):
        super(VerificationNetwork, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.base_network = base_network

    '''need to  repeat this method for each class so that it describes the distance between the corresponding class 
    and the closest other class'''

    def attach_property_layers(self, model: Net, true_class_index: int):
        n_classes = model.layers[-1].out_features
        cases = []
        for i in range(n_classes):
            if i == true_class_index:
                continue
            case = [0] * n_classes  # list of zeroes
            case[true_class_index] = 1  # sets the property to 1
            case[i] = -1
            cases.append(case)
        weights = np.array(cases)
        #         print(f'weight={weights}')
        weight_tensor = torch.from_numpy(weights).float().to(device)
        print(f'final weightTensor size={weight_tensor.size()}')
        return weight_tensor

    def forward_verif(self, x, true_class_index):
        x = self.base_network(x)
        property_layer = self.attach_property_layers(self.base_network, true_class_index)
        result = torch.matmul(x, torch.t(property_layer))
        return torch.min(result, dim=1, keepdim=True)[0]

    def forward(self, x):
        #         x = self.base_network(x)
        x = self.base_network(x)
        return x

    def get_upper_bound(self, domain, model, true_class_index):
        # we try get_upper_bound
        nb_samples = 1024
        nb_inp = domain.size()[2:]  # get last dimensions
        print(nb_inp)
        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        rand_samples_size = [self.batch_size, nb_samples] + list(nb_inp)
        rand_samples = torch.zeros(rand_samples_size).to(device)
        rand_samples.uniform_(0, 1)
        domain_lb = domain.select(0, 0).contiguous()
        domain_ub = domain.select(0, 1).contiguous()
        domain_width = domain_ub - domain_lb
        domain_lb = domain_lb.view([self.batch_size, 1] + list(nb_inp)).expand(
            [self.batch_size, nb_samples] + list(nb_inp))  # expand the initial point for the number of examples
        domain_width = domain_width.view([self.batch_size, 1] + list(nb_inp)).expand(
            [self.batch_size, nb_samples] + list(nb_inp))  # expand the width for the number of examples
        inps = domain_lb + domain_width * rand_samples
        # now flatten the first dimension into the second
        flattened_size = [inps.size(0) * inps.size(1)] + list(inps.size()[2:])
        # print(flattened_size)
        # rearrange the tensor so that is consumable by the model
        print(self.data_size)
        examples_data_size = [flattened_size[0]] + list(self.data_size[1:])  # the expected dimension of the example tensor
        # print(examples_data_size)
        var_inps = inps.view(examples_data_size)
        if var_inps.size() != self.data_size: print(f"var_inps != data_size , {var_inps}/{self.data_size}")  # should match data_size
        outs = model.forward_verif(var_inps.view(-1, 784), true_class_index)  # gets the input for the values
        print(outs.size())
        print(outs[0])  # those two should be very similar but different because they belong to two different examples
        print(outs[1])
        print(outs.size())
        outs_true_class_resized = outs.view(self.batch_size, nb_samples)
        print(outs_true_class_resized.size())  # resize outputs so that they each row is a different element of each batch
        upper_bound, idx = torch.min(outs_true_class_resized, dim=1)  # this returns the distance of the network output from the given class, it selects the class which is furthest from the current one
        print(f'idx size {idx.size()}')
        print(f'inps size {inps.size()}')
        print(idx[0])
        # upper_bound = upper_bound[0]
        unsqueezed_idx = idx.view(-1, 1)
        print(f'single size {inps[0][unsqueezed_idx[0][0]][:].size()}')
        ub_point = torch.tensor([inps[x][idx[x]][:].cpu().numpy() for x in range(idx.size()[0])]).to(device)  # ub_point represents the input that amongst all examples returns the minimum response for the appropriate class
        return ub_point, upper_bound

    def get_lower_bound(self, domain, model):
        '''
        input_domain: Tensor containing in each row the lower and upper bound
                      for the corresponding dimension
        '''
        # now try to do the lower bound
        import gurobipy as grb
        batch_size = domain.size()[1]
        for index in range(batch_size):
            input_domain = domain.select(1, index)  # we use a single domain, not ready for parallelisation yet
            print(f'input_domain.size()={input_domain.size()}')
            lower_bounds = []
            upper_bounds = []
            gurobi_vars = []
            # These three are nested lists. Each of their elements will itself be a
            # list of the neurons after a layer.

            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', False)
            gurobi_model.setParam('Threads', 1)

            ## Do the input layer, which is a special case
            inp_lb = []
            inp_ub = []
            inp_gurobi_vars = []
            for dim in range(input_domain.size()[1]):
                ub = input_domain[1][dim]  # check this value, it can be messed up
                lb = input_domain[0][dim]
                #     print(f'ub={ub} lb={lb}')
                assert ub > lb, "ub should be greater that lb"
                #     print(f'ub={ub} lb={lb}')
                v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                                        vtype=grb.GRB.CONTINUOUS,
                                        name=f'inp_{dim}')
                inp_gurobi_vars.append(v)
                inp_lb.append(lb)
                inp_ub.append(ub)
            gurobi_model.update()

            lower_bounds.append(inp_lb)
            upper_bounds.append(inp_ub)
            gurobi_vars.append(inp_gurobi_vars)

            # print(lower_bounds[0][0])
            # print(upper_bounds[0][0])

            # print(model.layers[0])
            # print(range(0))

            layer_idx = 1
            for layer in model.layers:
                print(f'layer_idx={layer_idx}')
                # layer = model.layers[0]
                new_layer_lb = []
                new_layer_ub = []
                new_layer_gurobi_vars = []
                if type(layer) is nn.Linear:
                    print(f'Linear')
                    for neuron_idx in range(layer.weight.size(0)):
                        if (layer.bias is None):
                            ub = 0
                            lb = 0
                            lin_expr = 0
                        else:
                            ub = layer.bias.data[neuron_idx]
                            lb = layer.bias.data[neuron_idx]
                            lin_expr = layer.bias.data[neuron_idx].item()  # adds the bias to the linear expression
                        #     print(f'bias_ub={ub} bias_lb={lb}')

                        for prev_neuron_idx in range(layer.weight.size(1)):
                            coeff = layer.weight.data[neuron_idx, prev_neuron_idx]  # picks the weight between the two neurons
                            #         print(f'coeff={coeff} upper={coeff*upper_bounds[-1][prev_neuron_idx]} lower={coeff*lower_bounds[-1][prev_neuron_idx]}')
                            #                 assert coeff*lower_bounds[-1][prev_neuron_idx]!=coeff*upper_bounds[-1][prev_neuron_idx], f"coeff={coeff} upper={coeff*upper_bounds[-1][prev_neuron_idx]} lower={coeff*lower_bounds[-1][prev_neuron_idx]}"
                            if coeff >= 0:
                                ub = ub + coeff * upper_bounds[-1][prev_neuron_idx]  # multiplies the ub
                                lb = lb + coeff * lower_bounds[-1][prev_neuron_idx]  # multiplies the lb
                            else:  # inverted
                                ub = ub + coeff * lower_bounds[-1][prev_neuron_idx]  # multiplies the ub
                                lb = lb + coeff * upper_bounds[-1][prev_neuron_idx]  # multiplies the lb
                            #         print(f'ub={ub} lb={lb}')
                            #                     assert ub!=lb
                            lin_expr = lin_expr + coeff.item() * gurobi_vars[-1][prev_neuron_idx]  # multiplies the unknown by the coefficient
                        #         print(lin_expr)
                        v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                                                vtype=grb.GRB.CONTINUOUS,
                                                name=f'lay{layer_idx}_{neuron_idx}')
                        gurobi_model.addConstr(v == lin_expr)
                        gurobi_model.update()
                        #     print(f'v={v}')
                        gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
                        gurobi_model.optimize()
                        #          print(f'gurobi status {gurobi_model.status}')
                        assert gurobi_model.status == 2, "LP wasn't optimally solved"
                        # We have computed a lower bound
                        lb = v.X
                        v.lb = lb

                        # Let's now compute an upper bound
                        gurobi_model.setObjective(v, grb.GRB.MAXIMIZE)
                        gurobi_model.update()
                        gurobi_model.reset()
                        gurobi_model.optimize()
                        assert gurobi_model.status == 2, "LP wasn't optimally solved"
                        ub = v.X
                        v.ub = ub

                        new_layer_lb.append(lb)
                        new_layer_ub.append(ub)
                        new_layer_gurobi_vars.append(v)
                elif type(layer) == nn.ReLU:
                    print('Relu')
                    for neuron_idx, pre_var in enumerate(gurobi_vars[-1]):
                        pre_lb = lower_bounds[-1][neuron_idx]
                        pre_ub = upper_bounds[-1][neuron_idx]

                        v = gurobi_model.addVar(lb=max(0, pre_lb),
                                                ub=max(0, pre_ub),
                                                obj=0,
                                                vtype=grb.GRB.CONTINUOUS,
                                                name=f'ReLU{layer_idx}_{neuron_idx}')
                        if pre_lb >= 0 and pre_ub >= 0:
                            # The ReLU is always passing
                            gurobi_model.addConstr(v == pre_var)
                            lb = pre_lb
                            ub = pre_ub
                        elif pre_lb <= 0 and pre_ub <= 0:
                            lb = 0
                            ub = 0
                            # No need to add an additional constraint that v==0
                            # because this will be covered by the bounds we set on
                            # the value of v.
                        else:
                            lb = 0
                            ub = pre_ub
                            gurobi_model.addConstr(v >= pre_var)

                            slope = pre_ub / (pre_ub - pre_lb)
                            bias = - pre_lb * slope
                            gurobi_model.addConstr(v <= slope * pre_var + bias)

                        new_layer_lb.append(lb)
                        new_layer_ub.append(ub)
                        new_layer_gurobi_vars.append(v)
                else:
                    raise NotImplementedError
                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)

                layer_idx += 1
            # Assert that this is as expected a network with a single output
            # assert len(gurobi_vars[-1]) == 1, "Network doesn't have scalar output"

            # last layer, minimise
            v = gurobi_model.addVar(lb=min(lower_bounds[-1]), ub=max(upper_bounds[-1]), obj=0,
                                    vtype=grb.GRB.CONTINUOUS,
                                    name=f'lay{layer_idx}_min')
            #     gurobi_model.addConstr(v == min(gurobi_vars[-1]))
            gurobi_model.addGenConstrMin(v, gurobi_vars[-1], name="minconstr")
            gurobi_model.update()
            #     print(f'v={v}')
            gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
            gurobi_model.optimize()

            gurobi_model.update()
            gurobi_vars.append([v])

            # We will first setup the appropriate bounds for the elements of the
            # input
            # is it just to be sure?
            for var_idx, inp_var in enumerate(gurobi_vars[0]):
                inp_var.lb = domain[0, 0, var_idx]
                inp_var.ub = domain[1, 0, var_idx]

            # We will make sure that the objective function is properly set up
            gurobi_model.setObjective(gurobi_vars[-1][0], grb.GRB.MINIMIZE)
            print(f'gurobi_vars[-1][0].size()={len(gurobi_vars[-1])}')
            # We will now compute the requested lower bound
            gurobi_model.update()
            gurobi_model.optimize()
            assert gurobi_model.status == 2, "LP wasn't optimally solved"
            print(f'gurobi status {gurobi_model.status}')
            print(f'Result={gurobi_vars[-1][0].X}')
            print(f'Result={gurobi_vars[-1]}')
            print(f'Result -1={gurobi_vars[-2]}')
            return gurobi_vars[-1][0].X
