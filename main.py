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

indexes_rand_samples = np.random.randint(low=1, high=100, size=rand_samples)


# Flatten and normalize images
num_pixels = X_train.shape[1] * X_train.shape[2]

print(num_pixels)


# Display Images
print(rand_samples // 2, ',' , 2)

for index, i in enumerate(indexes_rand_samples):
  plt.subplot(4, rand_samples//2, index+1)
  plt.tight_layout()
  plt.imshow(X_train[i], cmap='gray', interpolation='none')
  plt.title("{}".format(Y_train[i]))
  plt.xticks([])
  plt.yticks([])

# show the plot
plt.show()
