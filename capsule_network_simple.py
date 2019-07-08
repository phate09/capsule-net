import argparse
import sys

import torch
from torch.optim import Adam
from torchvision import datasets, transforms

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


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    device = torch.device("cuda" if use_cuda else "cpu")
    model = capsule_network.CapsuleNet().to(device)
    # model.load_state_dict(torch.load('save/mnist_caps_10.pt'))
    print("# parameters:", sum(param.numel() for param in model.parameters()))
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = Adam(model.parameters())
    capsule_loss = capsule_network.CapsuleLoss()
    for epoch in range(1, args.epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, labels = data.to(device), target.to(device)
            target = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels.to("cpu")).to(device)  # one hot encoding
            optimizer.zero_grad()
            # output = model(data)
            classes, reconstructions = model(data, target)
            loss = capsule_loss(data, target, classes, reconstructions)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
        torch.save(model.state_dict(), f"./save/mnist_caps_{epoch}.pt")
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "./save/mnist_caps.pt")


if __name__ == '__main__':
    main()
