from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt

from pprint import pprint

import numpy as np

import urllib
import random
import sys
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, help="Choose how many samples to retrive for train and test even number")
args = parser.parse_args()

rand_samples = 4

if args.samples:
  rand_samples = args.samples
  print("samples on")



# Download database from MNIST to train the model
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


#Get random samples from database to train
random_X_train = X_train[np.random.choice(X_train.shape[0], rand_samples, replace=False), :]

#Get random samples from database to test
random_X_test = X_test[np.random.choice(X_test.shape[0], rand_samples, replace=False), :]

print(rand_samples // 2, ',' , 2)

for i in range(rand_samples):
  plt.subplot(4, rand_samples//2, i+1)
  plt.tight_layout()
  plt.imshow(X_train[i], cmap='gray', interpolation='none')
  plt.title("{}".format(Y_train[i]))
  plt.xticks([])
  plt.yticks([])

# show the plot
plt.show()
