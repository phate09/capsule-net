import numpy as np
import torch.nn
import torch.nn

from plnn.simplified.conv_net2 import Net
from plnn.simplified.conv_net_convert import convert_conv2d, get_weights


def test1(): # DEPRECATED
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


if __name__ == '__main__':
    # test1() DEPRECATED
    test1_5()
    # test2()
    # test3()
