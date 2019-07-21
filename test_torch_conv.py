import numpy as np
import torch.nn
import torch.nn
import torch.nn.functional as F

from plnn.simplified.conv_net2 import Net
from plnn.simplified.conv_net_convert import convert_conv2d, get_weights


def test1():  # DEPRECATED
    """This test ensures that the conversion between conv2d and linear works, returned value should be 0.0
    """
    batch_size = 1
    y = np.random.rand(batch_size, 1, 28, 28)

    torch.set_default_dtype(torch.float64)  # forces to use double precision as default
    model = Net().to('cpu')
    model.load_state_dict(torch.load('save/conv_net.pt', map_location='cpu'))
    x = model.sequential[0]
    l = x(torch.from_numpy(y)).detach().numpy()

    input_size = y.shape[1:]
    weights, params, new_size = convert_conv2d(x.weight, x.bias, input_size)
    W, b = weights
    x2 = torch.nn.Linear(784, 3380)  # doesn't matter
    x2.weight.data = torch.from_numpy(W)
    x2.bias.data = torch.from_numpy(b)
    l2 = x2(torch.from_numpy(y).reshape(batch_size, -1)).detach().numpy()
    l2 = np.reshape(l2, (1, 26, 26, 5))  # .transpose(l, (0, 2, 3, 1))
    l2 = l2.transpose(0, 3, 1, 2)
    print(l.shape)
    print(l2.shape)
    print(np.abs(l2 - l).max())  # number should be negligible
    print(f'Test1 is {"NOT " if np.abs(l2 - l).max() > 1e-13 else ""}PASSED\n')


def test1_5():
    """This test ensures that the conversion between conv2d and linear works, returned value should be 0.0
    """
    batch_size = 1
    y = np.random.rand(batch_size, 1, 28, 28)

    torch.set_default_dtype(torch.float64)  # forces to use double precision as default
    model = Net().to('cpu')
    model.load_state_dict(torch.load('save/conv_net.pt', map_location='cpu'))
    x = model.sequential[0]
    l = x(torch.from_numpy(y)).detach().numpy()

    input_size = y.shape[1:]
    weights, params, new_size = convert_conv2d(x.weight, x.bias, input_size)
    W, b = weights
    x2 = torch.nn.Linear(784, 3380)  # doesn't matter
    x2.weight.data = torch.from_numpy(W)
    x2.bias.data = torch.from_numpy(b)
    l2 = x2(torch.from_numpy(y).reshape(batch_size, -1)).detach().numpy()
    l2 = np.reshape(l2, (1, 5, 26, 26))  # .transpose(l, (0, 2, 3, 1))
    # l2 = l2.transpose(0, 3, 1, 2)
    print(l.shape)
    print(l2.shape)
    print(np.abs(l2 - l).max())  # number should be negligible
    print(f'Test1.5 is {"NOT " if np.abs(l2 - l).max() > 1e-13 else ""}PASSED\n')


def test2():
    """This test ensures that fully connnected layers are retrieved correctly
    """
    batch_size = 128

    torch.set_default_dtype(torch.float64)  # forces to use double precision as default
    model = Net().to('cpu')
    model.load_state_dict(torch.load('save/conv_net.pt', map_location='cpu'))
    x = model.sequential[3]
    y = np.random.rand(batch_size, x.weight.size()[1])
    l = x(torch.from_numpy(y)).detach().numpy()

    input_size = (batch_size, x.weight.size()[1])
    weights, params = get_weights([x], input_size)
    W, b = weights[1]
    x2 = torch.nn.Linear(784, 3380)  # doesn't matter
    x2.weight.data = torch.from_numpy(W)
    x2.bias.data = torch.from_numpy(b)
    l2 = x2(torch.from_numpy(y).reshape(batch_size, -1)).detach().numpy()
    l2 = np.reshape(l2, l.shape)  # .transpose(l, (0, 2, 3, 1))
    print(l.shape)
    print(l2.shape)
    print(np.abs(l2 - l).max())  # number should be negligible
    print(f'Test2 is {"NOT " if np.abs(l2 - l).max() > 0 else ""}PASSED')


def test3():
    batch_size = 128
    y = np.random.rand(batch_size, 1, 28, 28)

    torch.set_default_dtype(torch.float64)  # forces to use double precision as default
    model = Net().to('cpu')
    model.load_state_dict(torch.load('save/conv_net.pt', map_location='cpu'))
    x = model.sequential
    l = x(torch.from_numpy(y)).detach().numpy()

    input_size = y.shape[1:]
    eq_weights, params = get_weights(model.layers, input_size)
    layers = []
    for i in range(len(eq_weights)):
        try:
            print(eq_weights[i][0].shape)
        except:
            continue
        out_features, in_features = eq_weights[i][0].shape
        layer = torch.nn.Linear(in_features, out_features)
        layer.weight.data = torch.from_numpy(eq_weights[i][0])
        layer.bias.data = torch.from_numpy(eq_weights[i][1])
        layers.append(layer)
        if i != len(eq_weights) - 1:
            layers.append(torch.nn.ReLU())
    x2 = torch.nn.Sequential(*layers)
    l2 = x2(torch.from_numpy(y).reshape(batch_size, -1)).detach().numpy()
    l2 = np.reshape(l2, l.shape)  # .transpose(l, (0, 2, 3, 1))
    print(l.shape)
    print(l2.shape)
    print(np.abs(l2 - l).max())  # number should be negligible
    print(f'Test3 is {"NOT " if np.abs(l2 - l).max() > 0 else ""}PASSED')


def test4():
    """test conversion of stride2 conv layer NOT WORKING"""
    batch_size = 128
    torch.set_default_dtype(torch.float64)  # forces to use double precision as default
    y = torch.tensor(np.random.rand(batch_size, 1, 20, 20))
    y = F.pad(y, (0, 1, 0, 1), 'constant', 0)
    x = torch.nn.Conv2d(1, 20, (9, 9), (2, 2))
    l = x(y).detach().numpy()
    input_size = y.shape[1:]
    # W = x.weight
    # b = x.bias
    weights, param, size = convert_conv2d(x.weight, x.bias, input_size, stride=(2, 2))
    W, b = weights[0], weights[1]
    lin1 = torch.nn.Linear(28 * 28, b.size)
    lin1.weight.data = torch.tensor(W, dtype=torch.double)
    lin1.bias.data = torch.tensor(b, dtype=torch.double)
    l2 = lin1(y.view(batch_size, -1)).detach()
    l2 = l2.reshape(l.shape).detach().numpy()
    print(l.shape)
    print(l2.shape)
    print(np.abs(l2 - l).max())  # number should be negligible
    print(f'Test3 is {"NOT " if np.abs(l2 - l).max() > 1e-13 else ""}PASSED')


def test5():
    """got the test from https://pytorch.org/docs/stable/nn.html#torch.nn.Unfold"""
    torch.set_default_dtype(torch.float64)
    inp = torch.randn(1, 3, 10, 12)
    w = torch.randn(2, 3, 4, 5)
    inp_unf = torch.nn.functional.unfold(inp, (4, 5))  # im2col operation
    out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
    # out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
    # or equivalently (and avoiding a copy),
    out = out_unf.view(1, 2, 7, 8)
    print((torch.nn.functional.conv2d(inp, w) - out).abs().max())


def test6():
    """testing the im2col (unfold in pytorch) algorithm for batch multiplication of convolutional layers"""
    torch.set_default_dtype(torch.float64)
    batch_size = 128
    inp = torch.randn(batch_size, 1, 28, 28)
    w = torch.randn(20, 1, 9, 9)
    inp_unf = torch.nn.functional.unfold(inp, (9, 9), stride=2)  # im2col operation
    out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
    # out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
    # or equivalently (and avoiding a copy),
    out = out_unf.view(128, 20, 10, 10)
    error = (torch.nn.functional.conv2d(inp, w, stride=2) - out).abs().max()
    print(error)
    print(f'Test3 is {"NOT " if error > 1e-13 else ""}PASSED')


def test7():
    """testing the im2col (unfold in pytorch) algorithm for second layer of caps net"""
    torch.set_default_dtype(torch.float64)
    batch_size = 1
    inp = torch.randn(batch_size, 256, 20, 20)
    w = torch.randn(32, 256, 9, 9)
    b = torch.ones(32)
    inp_unf = torch.nn.functional.unfold(inp, (9, 9), stride=1)  # im2col operation
    out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
    # out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
    # or equivalently (and avoiding a copy),
    out_unf = torch.add(out_unf, b[None,::,None])
    out = out_unf.view(1, 32, 12, 12)
    error = (torch.nn.functional.conv2d(inp, w, bias=b, stride=1) - out).abs().max()
    print(error)
    print(f'Test3 is {"NOT " if error > 1e-12 else ""}PASSED')


if __name__ == '__main__':
    # test1() DEPRECATED
    # test1_5()
    # test2()
    # test3()
    # test4() #not working, replacing with im2col
    # test5()
    test7()
