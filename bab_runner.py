from typing import List, Any

from plnn.mini_net import Net
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_domain(input_tensor, eps_size):
    return torch.stack((input_tensor - eps_size, input_tensor + eps_size))


model = Net()
model.load_state_dict(torch.load('save/mini_net.pt'))
model.to(device)
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


class VerificationNetwork(nn.Module):
    def __init__(self, base_network, true_class_index):
        super(VerificationNetwork, self).__init__()
        self.true_class_index = true_class_index
        #         self.property_layers=[]
        #         n_classes = base_network.layers[-1].out_features
        #         print(f'n_classes={n_classes}')
        #         for true_class_index in range(n_classes):
        #             self.property_layers.append( self.attach_property_layers(base_network,true_class_index))
        self.property_layer = self.attach_property_layers(base_network, self.true_class_index)
        self.layers = base_network.layers + [self.property_layer]
        self.out = nn.Sequential(*self.layers)

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
        print(f'property layer weights.size()={weights.shape}')
        weightTensor = nn.Linear(in_features=n_classes, out_features=n_classes - 1,
                                 bias=False)
        print(f'initial weightTensor size={weightTensor.weight.size()}')
        weightTensor.weight.data = torch.from_numpy(weights).float()
        print(f'final weightTensor size={weightTensor.weight.size()}')
        return weightTensor

    def forward(self, x):
        #         x = self.base_network(x)
        x = self.out(x)
        print(x)
        print(x.size())
        return torch.min(x, dim=1, keepdim=True)[0]


# get the data and label
data, target = next(iter(test_loader))
print(f'data size:{data.size()}')
# print(data[0])
# create the domain
domain_raw = generate_domain(data, 0.001)
data_size = data.size()
print(f'domain size:{domain_raw.size()}')
print(model.layers[-1].out_features)
print(f'True class={target}')
single_true_class = 7
verification_model = VerificationNetwork(model, single_true_class)
verification_model.to(device)
test_out = verification_model(data.to(device).view(-1, 784))
print(f'test_out={test_out}')
print(f'targets={target}')
# print(f'test_out[0]={test_out[0]}')
test_out = model(data.to(device).view(-1, 784))
print(test_out.size())
domain = domain_raw.view(2, batch_size, -1)
print(domain.size())


# global_ub_point, global_ub = net.get_upper_bound(domain)
# global_lb = net.get_lower_bound(domain)


def get_upper_bound(domain, model):
    # we try get_upper_bound
    nb_samples = 1024
    nb_inp = domain.size()[2:]  # get last dimensions
    print(nb_inp)
    # Not a great way of sampling but this will be good enough
    # We want to get rows that are >= 0
    rand_samples_size = [batch_size, nb_samples] + list(nb_inp)
    rand_samples = torch.zeros(rand_samples_size)
    rand_samples.uniform_(0, 1)
    domain_lb = domain.select(0, 0).contiguous()
    domain_ub = domain.select(0, 1).contiguous()
    domain_width = domain_ub - domain_lb
    domain_lb = domain_lb.view([batch_size, 1] + list(nb_inp)).expand(
        [batch_size, nb_samples] + list(nb_inp))  # expand the initial point for the number of examples
    domain_width = domain_width.view([batch_size, 1] + list(nb_inp)).expand(
        [batch_size, nb_samples] + list(nb_inp))  # expand the width for the number of examples
    # print(domain_width.size())
    # those should be the same
    # print(domain_width.size())
    # print(rand_samples.size())
    inps = domain_lb + domain_width * rand_samples
    # print(inps) #each row shuld be different
    # print(inps.size())
    # now flatten the first dimension into the second
    flattened_size = [inps.size(0) * inps.size(1)] + list(inps.size()[2:])
    # print(flattened_size)
    # rearrange the tensor so that is consumable by the model
    print(data_size)
    examples_data_size = [flattened_size[0]] + list(data_size[1:])  # the expected dimension of the example tensor
    # print(examples_data_size)
    var_inps = torch.Tensor(inps).view(examples_data_size)
    if var_inps.size() != data_size: print(f"var_inps != data_size , {var_inps}/{data_size}")
    print(f'var_inps.size()={var_inps.size()}')  # should match data_size
    print(inps.size())
    outs = model.forward(var_inps.to(device).view(-1, 784))  # gets the input for the values
    print(outs.size())
    print(outs[0])  # those two should be very similar but different because they belong to two different examples
    print(outs[1])
    print(target.unsqueeze(1))
    target_expanded = target.unsqueeze(1).expand(
        [batch_size, nb_samples])  # generates nb_samples copies of the target vector, all rows should be the same
    print(target_expanded.size())
    print(target_expanded)
    target_idxs = target_expanded.contiguous().view(
        batch_size * nb_samples)  # contains a list of indices that tells which columns out of the 10 classes to pick
    print(target_idxs.size())  # the first dimension should match
    print(outs.size())
    print(outs[target_idxs[0]].size())
    outs_true_class = outs.gather(1, target_idxs.to(device).view(-1,
                                                             1))  # we choose dimension 1 because it's the one we want to reduce
    print(outs_true_class.size())
    # print(outs[0])
    # print(target_idxs[1])
    # print(outs[1][0])#these two should be similar but different because they belong to different examples
    # print(outs[0][0])
    print(outs_true_class.size())
    outs_true_class_resized = outs_true_class.view(batch_size, nb_samples)
    print(outs_true_class_resized.size())  # resize outputs so that they each row is a different element of each batch
    upper_bound, idx = torch.min(outs_true_class_resized,
                                 dim=1)  # this returns the distance of the network output from the given class, it selects the class which is furthest from the current one
    print(upper_bound.size())
    print(idx.size())
    print(idx)
    print(upper_bound)
    # rearranged_idx=idx.view(list(inps.size()[0:2]))
    # print(rearranged_idx.size()) #rearranged idx contains the indexes of the minimum class for each example, for each element of the batch
    print(f'idx size {idx.size()}')
    print(f'inps size {inps.size()}')
    print(idx[0])
    # upper_bound = upper_bound[0]
    unsqueezed_idx = idx.to(device).view(-1, 1)
    print(f'single size {inps[0][unsqueezed_idx[0][0]][:].size()}')
    print(f'single size {inps[1][unsqueezed_idx[1][0]][:].size()}')
    print(f'single size {inps[2][unsqueezed_idx[2][0]][:].size()}')
    ub_point = [inps[x][idx[x]][:].numpy() for x in range(idx.size()[0])]
    ub_point = torch.tensor(ub_point)
    print(
        ub_point)  # ub_point represents the input that amongst all examples returns the minimum response for the appropriate class
    print(ub_point.size())
    # print(unsqueezed_idx.size())
    # ub_point = torch.gather(inps.to(device),1,unsqueezed_idx.to(device))#todo for some reason it doesn't want to work
    # print(ub_point.size())
    return ub_point, upper_bound


# test the method
get_upper_bound(domain, verification_model)
