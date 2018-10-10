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

print(math.ceil(rand_samples / 2), ',' , 2)
fig=plt.figure(figsize=(8, 6))

columns = 2
rows = math.ceil(rand_samples / 2)
aux = 220
for i in range(0, columns * rows):
  fig.add_subplot(rows, columns, i + 1, )
  #plt.subplot(aux + 1)
  plt.imshow(random_X_train[i], cmap=plt.get_cmap('gray'))
  print(i)
"""
plt.subplot(221)
plt.imshow(random_X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(random_X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(random_X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(random_X_train[3], cmap=plt.get_cmap('gray'))
"""
# show the plot
plt.show()
