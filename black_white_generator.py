import numpy as np


class BlackWhite:
    def __init__(self, shape):
        self.shape = shape

    def generate(self):
        if np.random.randint(0, 2) > 0:
            return np.ones(shape=self.shape)
        else:
            return np.zeros(shape=self.shape)
