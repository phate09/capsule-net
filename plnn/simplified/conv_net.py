from __future__ import print_function

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
import torch.utils.data as utils

from black_white_generator import BlackWhite
from plnn.flatten import Flatten


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # self.fc1 = nn.Linear(4 * 4 * 50, 500)
        # self.fc2 = nn.Linear(500, 10)
        self.layers = [
            nn.Conv2d(1, 5, 3),
            nn.ReLU(),
            nn.Conv2d(5, 5, 3),
            nn.ReLU(),
            nn.Conv2d(5, 5, 3),  # (22*22*5)
            nn.ReLU(),
            Flatten(),
            nn.Linear(2420, 2)
            # nn.Conv2d(1, 5, 3),
            # nn.ReLU(),
            # Flatten(),
            # nn.Linear(784,2)
        ]
        self.sequential = nn.Sequential(*self.layers)

    def forward(self, x):
        return F.log_softmax(self.sequential(x), dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()  # train mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # print(data.size())
        output = model(data)
        loss = F.nll_loss(output, target.long())
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, flatten=False):
    model.eval()  # evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.contiguous().view(128, -1) if flatten else data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    black_white = BlackWhite(shape=(1, 28, 28))

    my_dataset = utils.TensorDataset(black_white.data, black_white.target)  # create your datset
    my_dataloader = utils.DataLoader(my_dataset, batch_size=128, shuffle=True, drop_last=True)  # create your dataloader

    train_sequence(args, device, my_dataloader)

    # test_sequence(args, device, my_dataloader)


# def test_sequence(args, device, my_dataloader):
#     model = Net().to('cpu')
#     model.load_state_dict(torch.load('../../save/conv_net.pt', map_location='cpu'))
#     model.to(device)
#     test(args, model, device, my_dataloader, flatten=False)
#     eq_weights, new_params = get_weights(model.layers, inp_shape=(1, 28, 28))
#     layers = []
#     for i in range(len(eq_weights)):
#         try:
#             print(eq_weights[i][0].shape)
#         except:
#             continue
#         in_features, out_features = eq_weights[i][0].shape
#         layer = nn.Linear(in_features, out_features)
#
#         layer.weight.data = torch.tensor(eq_weights[i][0], dtype=torch.float)
#         layer.bias.data = torch.tensor(eq_weights[i][1], dtype=torch.float)
#         layers.append(layer)
#         if i != len(eq_weights) - 1:
#             layers.append(nn.ReLU())
#     sequential = nn.Sequential(*layers)
#     model.layers = layers
#     model.sequential = sequential
#     model.to(device)
#     # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#     test(None, model, device, my_dataloader, flatten=True)  # after conversion to FC layer


def train_sequence(args, device, my_dataloader):
    model = Net().to(device)
    # get_weights(model, inp_shape=(1, 28, 28))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, my_dataloader, optimizer, epoch)
        test(args, model, device, my_dataloader)
    if (args.save_model):
        torch.save(model.state_dict(), "../../save/conv_net.pt")


if __name__ == '__main__':
    main()
