import torch
import torch.utils.data
from torch import nn
from torchvision import datasets, transforms
from torchvision.datasets.mnist import MNIST

from plnn.branch_and_bound import bab
from plnn.mnist_basic import Net
from verification.verification_network import VerificationNetwork

use_cuda = True
device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")


def generate_domain(input_tensor, eps_size):
    return torch.stack((input_tensor - eps_size, input_tensor + eps_size), -1)


model = Net()
model.load_state_dict(torch.load('save/mnist_cnn.pt',map_location='cpu'))
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
# convL: torch.nn.Conv2d = verification_model.base_network.layers[0]
# fcl, output_size = verification_model.convert_ConvL_to_FCL(data, convL.weight, convL.padding[0], convL.stride[0])
# stretch_kernel = verification_model.stretchKernel(convL.weight)
# result: torch.Tensor = torch.matmul(fcl, stretch_kernel)
# result_final = result.transpose(0, 1) + convL.bias.unsqueeze(1).expand(-1, 576).cpu()
# result_reshaped = result_final.reshape(output_size)
# result2 = convL(data)
# im2col.im2col_indices(data.cpu().numpy(),)

epsilon = 1e-2
decision_bound = 0
successes = 0
attempts = 0
last_result = ""
for data, target in iter(test_loader):
    domain_raw = generate_domain(data, 0.001)
    domain = domain_raw.to(device)
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
