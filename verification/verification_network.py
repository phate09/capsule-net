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
        weightTensor = torch.from_numpy(weights).float().to(device)
        print(f'final weightTensor size={weightTensor.size()}')
        return weightTensor

    def forward_verif(self, x, true_class_index):
        x = self.base_network(x)
        property_layer = self.attach_property_layers(self.base_network, true_class_index)
        result = torch.matmul(x, torch.t(property_layer))
        return torch.min(result, dim=1, keepdim=True)[0]

    def forward(self, x):
        #         x = self.base_network(x)
        x = self.base_network(x)
        return x

    def get_upper_bound(self,domain, model, true_class_index):
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
