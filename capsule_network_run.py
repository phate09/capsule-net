import sys

import torch
from torchvision import datasets, transforms

import capsule_network

sys.setrecursionlimit(15000)

BATCH_SIZE = 100
NUM_CLASSES = 10
use_cuda = True

torch.manual_seed(0)


def test():
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=64, shuffle=True, **kwargs)
    device = torch.device("cuda" if use_cuda else "cpu")
    model = capsule_network.CapsuleNet().to(device)
    model.load_state_dict(torch.load('save/mnist_caps_10.pt'))
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            target = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels.to("cpu")).to(device)  # one hot encoding
            classes, reconstructions = model(data)
            pred = classes.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    test()
