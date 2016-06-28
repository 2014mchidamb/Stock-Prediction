# Adapted from Michael Nielsen's book

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
    x = np.array(x).reshape(-1, 1)
    for b, w in zip(self.biases, self.weights):
      # relu activation
      x = np.maximum((np.dot(w, x)+b), 0)
    return x

  def SGD(self, training_data, epochs, mini_batch_size, l_rate, lmbda):
    n = len(training_data)
    for i in xrange(epochs):
      random.shuffle(training_data)
      mini_batches = [training_data[k:k+mini_batch_size]
                      for k in xrange(0, n, mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, l_rate, lmbda, n)
      print "Training cost for epoch ", i, " is: ", self.total_cost(training_data, lmbda)

  def update_mini_batch(self, mini_batch, l_rate, lmbda, n):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    # adagrad update
    historical_grad = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
      delta_nabla_b, delta_nabla_w = self.backprop(x, y)
      nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
      historical_grad += np.square(nabla_w)
    self.weights = [(1-l_rate*(lmbda/n))*w-(l_rate/len(mini_batch))*nw/(0.000001+np.sqrt(hg))
                    for w, nw, hg in zip(self.weights, nabla_w, historical_grad)]
    self.biases = [b-(l_rate/len(mini_batch))*nb
                   for b, nb in zip(self.biases, nabla_b)]

  def backprop(self, x, y):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    # feedforward
    activation = np.array(x).reshape(-1, 1)
    activations = [activation]
    zs = []
    for w, b in zip(self.weights, self.biases):
      z = np.dot(w, activation)+b
      zs.append(z)
      # relu activation
      activation = np.maximum(z, 0)
      activations.append(activation)
    # backward pass
    delta = (activation-y)
    delta[z <= 0] = 0
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    for l in xrange(2, self.num_layers):
      z = zs[-l]
      delta = np.dot(self.weights[-l+1].transpose(), delta)
      delta[z <= 0] = 0
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)

  def total_cost(self, training_data, lmbda):
    total_cost = 0
    for x, y in training_data:
      total_cost += 0.5*np.linalg.norm(self.feedforward(x)-y)**2
    total_cost /= len(training_data)
    total_cost += 0.5*(lmbda/len(training_data))*sum(
        np.linalg.norm(w)**2 for w in self.weights)
    return total_cost

def standardize(training):
  x_vals = np.array([x for x, y in training])
  x_vals -= np.mean(x_vals, axis = 0)
  x_vals /= np.std(x_vals, axis = 0)
  return [(x, y[1]) for x, y in zip(x_vals, training)]
