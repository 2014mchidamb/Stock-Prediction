from data_collection import get_data, get_point
from neural_net import NeuralNetwork
import numpy as np

# A newline separated list of tickers
portfolio_file = open('portfolio.txt', 'r')
portfolio = portfolio_file.read().split('\n')
portfolio_file.close()

# Data parameters
start_date = '2014-01-01'
end_date = '2016-06-21'
num_days_per_point = 7
traits = ['Volume', 'High', 'Low', 'Close', 'Open']

for stock in portfolio:
  if len(stock) == 0:
    continue
  training_data = get_data(stock, start_date, end_date,
                           num_days_per_point, traits)
  net = NeuralNetwork([num_days_per_point*len(traits), 10, 1])
  net.SGD(training_data, 100, 10, 3.0, lmbda = 5.0)
  tomorrow_input = get_point(stock, '2016-06-13', '2016-06-21', traits)
  print net.feedforward(tomorrow_input)
