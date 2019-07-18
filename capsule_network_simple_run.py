import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import plnn.simplified.conv_net_convert as convert

import capsule_network

sys.setrecursionlimit(15000)

BATCH_SIZE = 100
NUM_CLASSES = 10
use_cuda = True

torch.manual_seed(0)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    capsule_loss = capsule_network.CapsuleLoss()
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            target = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels.to("cpu")).to(device)  # one hot encoding
            classes, reconstructions = model(data)
            test_loss = capsule_loss(data, target, classes, reconstructions)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = classes.max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred_one_hot = one_hot_encode(pred.squeeze(), NUM_CLASSES).to(device)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def one_hot_encode(data, num_classes=10):
    return torch.eye(num_classes).index_select(dim=0, index=data.to("cpu"))


def capsule_conversion(capsule, input_size=(1, 28, 28)):
    conv1 = capsule.conv1
    weights, param, size = convert.convert_conv2d(conv1.weight, conv1.bias, input_size)
    # weights = [np.random.random(80281600), np.random.random(102400)]
    # size = (102400,)
    W, b = weights[0], weights[1]
    lin1 = nn.Linear(28 * 28, b.size)
    lin1.weight.data = torch.tensor(W)
    lin1.bias.data = torch.tensor(b)
    print(lin1)
    convs_caps1 = [caps for caps in capsule.primary_capsules.capsules]
    size2 = (256,20,20)
    for conv in convs_caps1:
        weights, param, size = convert.convert_conv2d(conv.weight, conv.bias, size2)
    pass


def main():
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    device = torch.device("cuda" if use_cuda else "cpu")
    model = capsule_network.CapsuleNet().to(device)
    model.load_state_dict(torch.load('save/mnist_caps_10.pt'))
    print("# parameters:", sum(param.numel() for param in model.parameters()))
    capsule_conversion(model)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


if __name__ == '__main__':
    main()
