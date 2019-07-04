import numpy
import torch.nn
import torch.nn

from plnn.simplified.conv_net2 import Net
from plnn.simplified.conv_net_convert import convert_conv2d, get_weights


def test1():
    """This test ensures that the conversion between conv2d and linear works, returned value should be 0.0
    """
    batch_size = 128
    y = numpy.random.rand(batch_size, 1, 28, 28)

    torch.set_default_dtype(torch.float64)  # forces to use double precision as default
    model = Net().to('cpu')
    model.load_state_dict(torch.load('save/conv_net.pt', map_location='cpu'))
    x = model.sequential[0]
    l = x(torch.from_numpy(y)).detach().numpy()

    input_size = (1, 28, 28)
    weights, params, new_size = convert_conv2d(x.weight, x.bias, input_size)
    W, b = weights
    x2 = torch.nn.Linear(784, 3380)  # doesn't matter
    x2.weight.data = torch.from_numpy(W)
    x2.bias.data = torch.from_numpy(b)
    l2 = x2(torch.from_numpy(y).reshape(batch_size, -1)).detach().numpy()
    l2 = numpy.reshape(l, (batch_size, 5, 26, 26))  # .transpose(l, (0, 2, 3, 1))
    print(l.shape)
    print(l2.shape)
    print(numpy.abs(l2 - l).max())  # number should be negligible
    print(f'Test1 is {"NOT " if numpy.abs(l2 - l).max() > 0 else ""}PASSED')


def test2():
    """This test ensures that fully connnected layers are retrieved correctly
    """
    batch_size = 128

    torch.set_default_dtype(torch.float64)  # forces to use double precision as default
    model = Net().to('cpu')
    model.load_state_dict(torch.load('save/conv_net.pt', map_location='cpu'))
    x = model.sequential[3]
    y = numpy.random.rand(batch_size, x.weight.size()[1])
    l = x(torch.from_numpy(y)).detach().numpy()

    input_size = (batch_size, x.weight.size()[1])
    weights, params = get_weights([x], input_size)
    W, b = weights[1]
    x2 = torch.nn.Linear(784, 3380)  # doesn't matter
    x2.weight.data = torch.from_numpy(W)
    x2.bias.data = torch.from_numpy(b)
    l2 = x2(torch.from_numpy(y).reshape(batch_size, -1)).detach().numpy()
    l2 = numpy.reshape(l2, l2.shape)  # .transpose(l, (0, 2, 3, 1))
    print(l.shape)
    print(l2.shape)
    print(numpy.abs(l2 - l).max())  # number should be negligible
    print(f'Test1 is {"NOT " if numpy.abs(l2 - l).max() > 0 else ""}PASSED')


if __name__ == '__main__':
    # test1()
    test2()
