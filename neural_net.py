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

  def feedforward(self, x):
    for b, w in zip(self.biases, self.weights):
      x = relu(np.dot(w, x)+b)
    return x

  def SGD(self, training_data, epochs, mini_batch_size, l_rate, lmda):
    n = len(training_data)
    for i in xrange(epochs):
      random.shuffle(training_data)
      mini_batches = [training_data[k:k+mini_batch_size]
                      for k in xrange(0, n, mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, l_rate, lmda, n)

  def update_mini_batch(self, mini_batch, l_rate, lmbda, n):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
      delta_nabla_b, delta_nabla_w = self.backprop(x, y)
      nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.weights =
