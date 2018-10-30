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

rand_samples = 12

if args.samples:
  rand_samples = args.samples
  print("samples on")



# Download database from MNIST to train the model
(digits_train, digits_titles_train), (digits_test, digits_titles_test) = mnist.load_data()

indexes_rand_samples = np.random.randint(low=1, high=100, size=rand_samples)

randomized_digits_train = [digits_train[i] for index, i in enumerate(indexes_rand_samples)]
randomized_digits_titles_train =[digits_titles_train[i] for index, i in enumerate(indexes_rand_samples)]

#print(randomized_digits_train)

# Flatten and normalize images 28x28
num_pixels = digits_train.shape[1] * digits_train.shape[2]

digits_train = digits_train.reshape(digits_train.shape[0], num_pixels).astype('float32')
digits_test = digits_test.reshape(digits_test.shape[0], num_pixels).astype('float32')


# Display Images
print(rand_samples // 2, ',' , 2)
plt.figure(figsize=(10,10))

for i in range(rand_samples):
  print(i)
  plt.subplot(4, rand_samples//2, i+1)
  plt.tight_layout()
  plt.imshow(randomized_digits_train[i], cmap='gray', interpolation='none')
  plt.title("{}".format(randomized_digits_titles_train[i]))
  plt.xticks([])
  plt.yticks([])

# show the plot
plt.show()
