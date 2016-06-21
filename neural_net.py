import numpy as np
import random

class NeuralNetwork(object):

  def __init__(self, sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.default_weight_initializer()

  def default_weight_initializer(self):
    # only non-input layers have biases
    self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
    self.weights = [np.random.randn(y, x)/np.sqrt(x)
                    for x, y in zip(self.sizes[:-1], self.sizes[1:])]
