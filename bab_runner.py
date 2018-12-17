#!/usr/bin/env python
import argparse

from plnn.branch_and_bound import bab
from plnn.network_linear_approximation import LinearizedNetwork
from plnn.model import load_and_simplify
from capsule_network import CapsuleNet, CapsuleLoss
import torch

BATCH_SIZE = 200
NUM_CLASSES = 10
def attachPropertyLayers(model: MnistClassifier, true_class_index: int):
  # allows to do y>y1,y2,y3... which is 'is it true that the output y corresponds to the true class with maximum response?'
  n_classes = model.y.get_shape().as_list()[1]
  cases = []
  for i in range(n_classes):
    if (i == true_class_index):
      continue
    case = [0] * n_classes  # list of zeroes
    case[true_class_index] = 1  # sets the property to 1
    case[i] = -1
    cases.append(case)
  layers = []
  with model.graph.as_default():
    weights = np.array(cases)
    weightTensor=tf.constant(np.rot90(weights), name='propertyLayer', dtype=tf.float32)
    all_properties_together = HiddenLayer(n_classes, n_classes-1,weightTensor=weightTensor,use_bias=False)#initialise the layer with constant weights
    # for case in cases:
    #   property_layer = PropertyLayer(case, model.y)
    #   layers.append(property_layer.output)
    # all_properties_together = tf.concat(layers, axis=0)
    layers.append(all_properties_together)
    forward = all_properties_together.forward(model.y)
    min_layer = tf.reduce_min(forward,axis=1)
    layers.append(min_layer)
  return min_layer, layers

def main():
    print("start")
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchnet.engine import Engine
    from torchnet.logger import VisdomPlotLogger, VisdomLogger
    from torchvision.utils import make_grid
    from torchvision.datasets.mnist import MNIST
    from tqdm import tqdm
    import torchnet as tnt
    print("capsulenet")
    model = CapsuleNet()
    model.load_state_dict(torch.load('epochs/epoch_500.pt'))
    model.cuda()
    def get_iterator(mode):
        dataset = MNIST(root='./data', download=True, train=mode)
        data = getattr(dataset, 'train_data' if mode else 'test_data')
        labels = getattr(dataset, 'train_labels' if mode else 'test_labels')
        tensor_dataset = tnt.dataset.TensorDataset([data, labels])

        return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=4, shuffle=mode)
    total_accuracy=0
    n_batches = 0
    for i_batch, sample_batched in enumerate(get_iterator(False)): #false = test mode
        data, labels = sample_batched
        ground_truth = (data.unsqueeze(1).float() / 255.0)
        labels = torch.LongTensor(labels)
        labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)
        ground_truth = Variable(ground_truth).cuda()
        labels = Variable(labels).cuda()
        classes, reconstructions = model(ground_truth)
        accuracy = (classes.max(dim=1)[1]==labels.max(dim=1)[1]).sum().item()/len(sample_batched[0])
        total_accuracy = total_accuracy+accuracy
        n_batches=n_batches+1
        # loss = capsule_loss(ground_truth, labels, classes, reconstructions)
        print(f'The accuracy is {accuracy*100}%')
    print(f'Total accuracy is {total_accuracy/n_batches}')
    # parser = argparse.ArgumentParser(description="Read a .rlv file"
    #                                  "and prove its property.")
    # parser.add_argument('rlv_infile', type=argparse.FileType('r'),
    #                     help='.rlv file to prove.')
    # args = parser.parse_args()
    #
    network, domain = load_and_simplify(args.rlv_infile,
                                        LinearizedNetwork)
    #
    # epsilon = 1e-2
    # decision_bound = 0
    # min_lb, min_ub, ub_point = bab(network, domain,
    #                                epsilon, decision_bound)
    #
    # if min_lb >= 0:
    #     print("UNSAT")
    # elif min_ub < 0:
    #     print("SAT")
    #     print(ub_point)
    # else:
    #     print("Unknown")


if __name__ == '__main__':
    main()
