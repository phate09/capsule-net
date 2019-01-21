from plnn.mnist_basic import Net
import torch
import torch.nn as nn
from torchvision.datasets.mnist import MNIST
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms


#
# def attachPropertyLayers(model: MnistClassifier, true_class_index: int):
#   # allows to do y>y1,y2,y3... which is 'is it true that the output y corresponds to the true class with maximum response?'
#   n_classes = model.y.get_shape().as_list()[1]
#   cases = []
#   for i in range(n_classes):
#     if (i == true_class_index):
#       continue
#     case = [0] * n_classes  # list of zeroes
#     case[true_class_index] = 1  # sets the property to 1
#     case[i] = -1
#     cases.append(case)
#   layers = []
#   with model.graph.as_default():
#     weights = np.array(cases)
#     weightTensor = tf.constant(np.rot90(weights), name='propertyLayer', dtype=tf.float32)
#     all_properties_together = HiddenLayer(n_classes, n_classes - 1, weightTensor=weightTensor,
#                                           use_bias=False)  # initialise the layer with constant weights
#     # for case in cases:
#     #   property_layer = PropertyLayer(case, model.y)
#     #   layers.append(property_layer.output)
#     # all_properties_together = tf.concat(layers, axis=0)
#     layers.append(all_properties_together)
#     forward = all_properties_together.forward(model.y)
#     min_layer = tf.reduce_min(forward, axis=1)
#     layers.append(min_layer)
#   return min_layer, layers


def generate_domain(input_tensor, eps_size):
    return torch.stack((input_tensor - eps_size, input_tensor + eps_size))


class VerificationNetwork(nn.Module):
    def __init__(self, in_features, out_features, base_network, out_function, true_class_index):
        super(VerificationNetwork, self).__init__()
        self.true_class_index = true_class_index
        self.out_function = out_function
        self.base_network = base_network
        self.out_features = out_features
        self.in_features = in_features
        self.property_layer = None
        self.attach_property_layers(self.base_network, self.true_class_index)

    def attach_property_layers(self, model: Net, true_class_index: int):
        n_classes = model.fc2.out_features
        cases = []
        for i in range(n_classes):
            if i == true_class_index:
                continue
            case = [0] * n_classes  # list of zeroes
            case[true_class_index] = 1  # sets the property to 1
            case[i] = -1
            cases.append(case)
        layers = []
        weights = np.array(cases)
        weightTensor = nn.Linear(in_features=n_classes, out_features=n_classes,
                                 bias=False)  # tf.constant(np.rot90(weights), name='propertyLayer', dtype=tf.float32)
        weightTensor.weight.data = torch.from_numpy(weights)
        # all_properties_together = HiddenLayer(n_classes, n_classes-1,weightTensor=weightTensor,use_bias=False)#initialise the layer with constant weights
        # for case in cases:
        #   property_layer = PropertyLayer(case, model.y)
        #   layers.append(property_layer.output)
        # all_properties_together = tf.concat(layers, axis=0)
        layers.append(weightTensor)
        # forward = all_properties_together.forward(model.y)
        min_layer = torch.min(weightTensor, axis=1)
        layers.append(min_layer)
        return min_layer, layers

    def forward(self, x):
        x = self.base_network(x)
        x = self.property_layer(x)
        return torch.min(x)

    def get_upper_bound(self, domain):
        '''
        Compute an upper bound of the minimum of the network on `domain`

        Any feasible point is a valid upper bound on the minimum so we will
        perform some random testing.
        '''
        nb_samples = 1024
        nb_inp = domain.size(0)
        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        rand_samples = torch.Tensor(nb_samples, nb_inp)
        rand_samples.uniform_(0, 1)

        domain_lb = domain.select(1, 0).contiguous()
        domain_ub = domain.select(1, 1).contiguous()
        domain_width = domain_ub - domain_lb

        domain_lb = domain_lb.view(1, nb_inp).expand(nb_samples, nb_inp)
        domain_width = domain_width.view(1, nb_inp).expand(nb_samples, nb_inp)

        inps = domain_lb + domain_width * rand_samples

        var_inps = Variable(inps, volatile=True)
        outs = self.net(var_inps)

        upper_bound, idx = torch.min(outs.data, dim=0)

        upper_bound = upper_bound[0]
        ub_point = inps[idx].squeeze()

        return ub_point, upper_bound

    def get_lower_bound(self, domain):
        '''
        Update the linear approximation for `domain` of the network and use it
        to compute a lower bound on the minimum of the output.

        domain: Tensor containing in each row the lower and upper bound for
                the corresponding dimension
        '''
        self.define_linear_approximation(domain)
        return self.compute_lower_bound(domain)


def main():
    model = Net()
    model.load_state_dict(torch.load('save/mnist_cnn.pt'))
    model.cuda()
    dataset = MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),  # load the testing dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)#retrieve items 1 at a time
    seed = 0
    for data, target in test_loader:
        # create the domain
        # print(data)
        # domain = generate_domain(data[0][0],0.001)
        # print(domain)
        print(model(data.cuda()))
        break


if __name__ == '__main__':
    main()
