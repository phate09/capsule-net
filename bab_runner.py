import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.datasets.mnist import MNIST

from plnn.branch_and_bound import bab
from plnn.mini_net import Net
from verification.verification_network import VerificationNetwork

use_cuda = False
device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")


def generate_domain(input_tensor, eps_size):
    return torch.stack((input_tensor - eps_size, input_tensor + eps_size), -1)


model = Net()
model.load_state_dict(torch.load('save/mini_net.pt', map_location='cpu'))
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

verification_model = VerificationNetwork(model)
verification_model.to(device)

epsilon = 1e-5
decision_bound = 0
successes = 0
attempts = 0
last_result = ""
max_samples =1000
for data, target in iter(test_loader):
    domain_raw = generate_domain(data, 1e-1)
    domain = domain_raw.to(device)  # at this point is (batch channel, width, height, bound)
    min_lb, min_ub, ub_point = bab(verification_model, domain, target.item(), epsilon, decision_bound,save=False)
    attempts += 1
    if min_lb >= 0:
        successes += 1
        last_result = "UNSAT"
    elif min_ub < 0:
        last_result = "SAT"
        # print(ub_point)
    else:
        print("Unknown")  # 18
    print(f'\rRunning percentage: {successes / attempts:.02%}, {attempts}/{min(len(test_loader),max_samples)}, last result:{last_result}', end="   ")
    if attempts >=max_samples:
        break
print(f'Final percentage: {successes / attempts:.02%}')
# 1e-1 = 90.10%
# 1e-2 = 91.40%
# 1e-3 = 91.60%
# 1e-4 = 91.60%
# 1e-5 = 91.60%
