import numpy
import tensorflow
import torch.nn
import torch.nn

from plnn.simplified.conv_net import Net
from plnn.simplified.conv_net_convert import convert_conv2d
'''This test shows that the conversion between tensorflow and pytorch works as expected, difference should be negligible'''

y = numpy.random.rand(1, 28, 28, 1)
filterx = numpy.random.rand(4, 5, 1, 2)  # W H I O
a = tensorflow.nn.conv2d(
    y,
    filterx, [1, 1, 1, 1], 'VALID')
with tensorflow.Session() as sess:
    t = (sess.run(a))

x = torch.nn.Conv2d(1, 10, 5, bias=False)
filter = numpy.transpose(filterx, (3, 2, 0, 1))  # O I W H
x.weight = torch.nn.Parameter(torch.from_numpy(filter))
z = numpy.transpose(y, (0, 3, 1, 2))
l = x(torch.from_numpy(z))
l = l.detach().numpy()
l = numpy.transpose(l, (0, 2, 3, 1))

print(t.shape)
print(l.shape)
print(numpy.abs(t - l).max())

torch.set_default_dtype(torch.float64)  # forces to use double precision as default
model = Net().to('cpu')
model.load_state_dict(torch.load('save/conv_net.pt', map_location='cpu'))
x = model.sequential[0]
# x.bias = None
a = tensorflow.nn.conv2d(
    y,
    x.weight.data.cpu().numpy().transpose(2, 3, 1, 0), [1, 1, 1, 1], 'VALID')
a = tensorflow.nn.bias_add(a, x.bias.data.cpu().numpy())
with tensorflow.Session() as sess:
    t = (sess.run(a))
z = numpy.transpose(y, (0, 3, 1, 2))
l = x(torch.from_numpy(z)).detach().numpy()
l = numpy.transpose(l, (0, 2, 3, 1))
print(t.shape)
print(l.shape)
print(numpy.abs(t - l).max())  # number should be negligible


input_size = (1, 28, 28)
weights, params,new_size = convert_conv2d(x.weight, x.bias,input_size )
W, b = weights
x = torch.nn.Linear(784, 3380)  # doesn't matter
x.weight.data = torch.from_numpy(W)
x.bias.data = torch.from_numpy(b)
l = x(torch.from_numpy(z).flatten()).detach().numpy()
l = numpy.reshape(l, (1, 26, 26, 5))  # .transpose(l, (0, 2, 3, 1))
print(t.shape)
print(l.shape)
print(numpy.abs(t - l).max())  # number should be negligible
