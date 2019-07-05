from __future__ import print_function

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torch.utils.data as utils

from black_white_generator import BlackWhite
from plnn.simplified.conv_net import Net


def get_weights(layers, inp_shape=(1, 28, 28)):
    temp_weights = [(layer.weight, layer.bias) if hasattr(layer, 'weight') else [] for layer in layers]
    new_params = []
    eq_weights = []
    cur_size = inp_shape
    for p in temp_weights:
        if len(p) > 0:
            W, b = p
            eq_weights.append([])
            if len(W.shape) == 2:  # FC
                eq_weights.append([W.data.cpu().numpy(), b.data.cpu().numpy()])
            else:  # Conv
                weights, param, new_size = convert_conv2d(W, b, cur_size)
                eq_weights.append(weights)
                new_params.append(param)
                cur_size = new_size
    print('Weights found')
    return eq_weights, new_params


def convert_conv2d(W, b, cur_size=(1, 28, 28)):  # works for pytorch input
    new_params = []
    eq_weights = []
    new_size = (W.shape[0], cur_size[-2] - W.shape[-2] + 1, cur_size[-1] - W.shape[-1] + 1)
    flat_inp = np.prod(cur_size)  # m x n
    flat_out = np.prod(new_size)  #
    new_params.append(flat_out)
    W_flat = np.zeros((flat_out, flat_inp))
    b_flat = np.zeros((flat_out))
    in_channel, in_height, in_width = cur_size
    o_channel, o_height, o_width = new_size
    for o_h in range(o_height):
        for o_w in range(o_width):
            for o_c in range(o_channel):
                b_flat[o_width * o_height * o_c + o_width * o_h + o_w] = b[o_c]
                for k in range(in_channel):
                    for idx0 in range(W.shape[2]):
                        for idx1 in range(W.shape[3]):
                            i = idx0 + o_h
                            j = idx1 + o_w
                            W_flat[o_width * o_height * o_c + o_width * o_h + o_w, in_width * in_height * k + in_width * i + j] = W[o_c, k, idx0, idx1]
    eq_weights.append([W_flat, b_flat])
    return eq_weights[0], new_params[0], new_size


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


def convert(input_layers):
    eq_weights, new_params = get_weights(input_layers, inp_shape=(1, 28, 28))  # pytorch style
    layers = []
    for i in range(len(eq_weights)):
        try:
            print(eq_weights[i][0].shape)
        except:
            continue
        out_features, in_features = eq_weights[i][0].shape
        layer = nn.Linear(in_features, out_features)
        layer.weight.data = torch.from_numpy(eq_weights[i][0].astype(np.float32))
        layer.bias.data = torch.from_numpy(eq_weights[i][1].astype(np.float32))
        layers.append(layer)
        if i != len(eq_weights) - 1:
            layers.append(nn.ReLU())
    return layers


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
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
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    black_white = BlackWhite(shape=(1, 28, 28))  # pytorch style

    my_dataset = utils.TensorDataset(black_white.data, black_white.target)  # create your datset
    my_dataloader = utils.DataLoader(my_dataset, batch_size=128, shuffle=True, drop_last=True)  # create your dataloader

    model = Net().to('cpu')
    model.load_state_dict(torch.load('../../save/conv_net.pt', map_location='cpu'))
    model.to(device)
    test(args, model, device, my_dataloader, flatten=False)
    layers = convert(model.layers)
    sequential = nn.Sequential(*layers)
    model.layers = layers
    model.sequential = sequential
    model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    test(args, model, device, my_dataloader, flatten=True)


if __name__ == '__main__':
    main()
