from plnn.branch_and_bound import CandidateDomain, bab
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
    return torch.stack((input_tensor - eps_size, input_tensor + eps_size), -1)


model = Net()
model.load_state_dict(torch.load('save/mini_net.pt'))
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
verification_model = VerificationNetwork(model, batch_size, data_size)
verification_model.to(device)
test_out = verification_model(data.view(-1, 784))
print(f'test_out={test_out}')
print(f'targets={target}')
# print(f'test_out[0]={test_out[0]}')
test_out = model(data.view(-1, 784))
print(test_out.size())
domain = domain_raw.view(-1, 2)
print(domain.size())

# test the method
# verification_model.get_upper_bound(domain, 7)

# test the method
# verification_model.get_lower_bound(domain, 7)

epsilon = 1e-2
decision_bound = 0
min_lb, min_ub, ub_point = bab(verification_model, domain, 7, epsilon, decision_bound)

if min_lb >= 0:
    print("UNSAT")
elif min_ub < 0:
    print("SAT")
    print(ub_point)
else:
    print("Unknown")
