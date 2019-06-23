import numpy as np
import torch


class BlackWhite:
    def __init__(self, shape=(1, 20, 20)):
        self.shape = shape
        my_x = [np.ones(shape, dtype=float), np.zeros(self.shape, dtype=float)]  # a list of numpy arrays
        my_y = [1, 0]  # another list of numpy arrays (targets)
        tensor_x: torch.Tensor = torch.stack([torch.Tensor(i) for i in my_x])  # transform to torch tensors
        tensor_x = tensor_x.repeat(20000, 1, 1, 1)
        tensor_y = torch.stack([torch.tensor(i) for i in my_y])
        tensor_y = tensor_y.repeat(20000)
        self.data = tensor_x
        self.target = tensor_y
