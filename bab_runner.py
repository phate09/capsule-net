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
domain_raw = generate_domain(data, 0.001)
data_size = data.size()
verification_model = VerificationNetwork(model, batch_size, data_size)
verification_model.to(device)

epsilon = 1e-2
decision_bound = 0
successes = 0
attempts = 0
last_result = ""
for data, target in iter(test_loader):
    domain_raw = generate_domain(data, 0.001)
    domain = domain_raw.view(-1, 2).to(device)
    min_lb, min_ub, ub_point = bab(verification_model, domain, target.item(), epsilon, decision_bound)
    attempts += 1
    if min_lb >= 0:
        successes += 1
        last_result = "UNSAT"
    elif min_ub < 0:
        last_result = "SAT"
        # print(ub_point)
    else:
        print("Unknown")  # 18
    print(f'\rRunning percentage: {successes / attempts:.02%}, {attempts}/{len(test_loader)}, last result:{last_result}', end="")
print(f'Final percentage: {successes / attempts:.02%}')
