from plnn.mini_net import Net
import torch
import torch.nn as nn
from torchvision.datasets.mnist import MNIST
import torch.utils.data
from torchvision import datasets, transforms

from verification.verification_network import VerificationNetwork

use_cuda = True
device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")


def generate_domain(input_tensor, eps_size):
    return torch.stack((input_tensor - eps_size, input_tensor + eps_size))


model = Net()
model.load_state_dict(torch.load('save/mini_net.pt'))
model
dataset = MNIST('./data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),  # load the testing dataset
batch_size = 1
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=False, pin_memory=True)
# test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)#retrieve items 1 at a time
seed = 0

# get the data and label
data, target = next(iter(test_loader))
data = data.to(device)
target = target.to(device)
print(f'data size:{data.size()}')
# print(data[0])
# create the domain
domain_raw = generate_domain(data, 0.001)
data_size = data.size()
print(f'domain size:{domain_raw.size()}')
print(model.layers[-1].out_features)
print(f'True class={target}')
single_true_class = 7
verification_model = VerificationNetwork(model)
verification_model.to(device)
test_out = verification_model(data.view(-1, 784))
print(f'test_out={test_out}')
print(f'targets={target}')
# print(f'test_out[0]={test_out[0]}')
test_out = model(data.view(-1, 784))
print(test_out.size())
domain = domain_raw.view(2, batch_size, -1)
print(domain.size())


# global_ub_point, global_ub = net.get_upper_bound(domain)
# global_lb = net.get_lower_bound(domain)





# test the method
VerificationNetwork.get_upper_bound(domain, verification_model, 7)

def get_lower_bound(domain,model):
    '''
    input_domain: Tensor containing in each row the lower and upper bound
                  for the corresponding dimension
    '''
    #now try to do the lower bound
    import gurobipy as grb
    batch_size=domain.size()[1]
    for index in range(batch_size):
        input_domain=domain.select(1,index)#we use a single domain, not ready for parallelisation yet
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
            ub=input_domain[1][dim]#check this value, it can be messed up
            lb=input_domain[0][dim]
        #     print(f'ub={ub} lb={lb}')
            assert ub>lb , "ub should be greater that lb"
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
                    if(layer.bias is None):
                        ub = 0
                        lb = 0
                        lin_expr = 0
                    else:
                        ub = layer.bias.data[neuron_idx]
                        lb = layer.bias.data[neuron_idx]
                        lin_expr = layer.bias.data[neuron_idx].item() #adds the bias to the linear expression
                #     print(f'bias_ub={ub} bias_lb={lb}')

                    for prev_neuron_idx in range(layer.weight.size(1)):
                        coeff = layer.weight.data[neuron_idx, prev_neuron_idx]#picks the weight between the two neurons
                #         print(f'coeff={coeff} upper={coeff*upper_bounds[-1][prev_neuron_idx]} lower={coeff*lower_bounds[-1][prev_neuron_idx]}')
        #                 assert coeff*lower_bounds[-1][prev_neuron_idx]!=coeff*upper_bounds[-1][prev_neuron_idx], f"coeff={coeff} upper={coeff*upper_bounds[-1][prev_neuron_idx]} lower={coeff*lower_bounds[-1][prev_neuron_idx]}"
                        if coeff>=0:
                            ub = ub+ coeff*upper_bounds[-1][prev_neuron_idx]#multiplies the ub
                            lb = lb+ coeff*lower_bounds[-1][prev_neuron_idx]#multiplies the lb
                        else: #inverted
                            ub = ub+ coeff*lower_bounds[-1][prev_neuron_idx]#multiplies the ub
                            lb = lb+ coeff*upper_bounds[-1][prev_neuron_idx]#multiplies the lb
                #         print(f'ub={ub} lb={lb}')
    #                     assert ub!=lb
                        lin_expr = lin_expr+ coeff.item() * gurobi_vars[-1][prev_neuron_idx]#multiplies the unknown by the coefficient
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

        #last layer, minimise
        v = gurobi_model.addVar(lb=min(lower_bounds[-1]), ub=max(upper_bounds[-1]), obj=0,
                                              vtype=grb.GRB.CONTINUOUS,
                                              name=f'lay{layer_idx}_min')
    #     gurobi_model.addConstr(v == min(gurobi_vars[-1]))
        gurobi_model.addGenConstrMin(v, gurobi_vars[-1], name= "minconstr")
        gurobi_model.update()
    #     print(f'v={v}')
        gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
        gurobi_model.optimize()

        gurobi_model.update()
        gurobi_vars.append([v])

        # We will first setup the appropriate bounds for the elements of the
        # input
        #is it just to be sure?
        for var_idx, inp_var in enumerate(gurobi_vars[0]):
            inp_var.lb = domain[0,0,var_idx]
            inp_var.ub = domain[1,0,var_idx]

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

#test the method
get_lower_bound(domain, verification_model, 7)