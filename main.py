from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

from pprint import pprint

import numpy as np

import sys
import os
import argparse
import math

from network import model

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, help="Choose how many samples to retrive for train and test even number * 100")
args = parser.parse_args()

rand_samples = 20*100

if args.samples:
  rand_samples = args.samples
  print("samples on")

# Grid Plot
print("Grid:", rand_samples // 2, ',' , 2)

# Download database from MNIST to train the model
(digits_train, digits_titles_train), (digits_test, digits_titles_test) = mnist.load_data()

# Create array of n number of random indices to choose
indexes_rand_samples = np.random.randint(low=1, high=100, size=rand_samples)

# Get random images from train dataset
randomized_digits_train = np.array([digits_train[i] for index, i in enumerate(indexes_rand_samples)])
randomized_digits_titles_train = np.array([digits_titles_train[i] for index, i in enumerate(indexes_rand_samples)])

# Get random images from test dataset
randomized_digits_test = np.array([digits_test[i] for index, i in enumerate(indexes_rand_samples)])
randomized_digits_titles_test = np.array([digits_titles_test[i] for index, i in enumerate(indexes_rand_samples)])

# Flatten and normalize pixel images to 0 and 1 on train dataset
num_pixels = randomized_digits_train.shape[1] * randomized_digits_train.shape[2]
digits_train = randomized_digits_train.reshape(randomized_digits_train.shape[0], num_pixels).astype('float32')
digits_train /= 255

# Flatten and normalize pixel images to 0 and 1 on test dataset
num_pixels = randomized_digits_test.shape[1] * randomized_digits_test.shape[2]
digits_test = randomized_digits_test.reshape(randomized_digits_test.shape[0], num_pixels).astype('float32')
digits_test /= 255

# Categorize the title digit numbers using one-hot encoding from 0 to 9
# For example, the nmber 4 should be represented as:
#   [0,0,0,1,0,0,0,0,0,0]

# Encode categories from 0 to 9 using one-hot encoding
print("Encoding categories train titles")
encoded_digits_titles_train = np_utils.to_categorical(randomized_digits_titles_train, 10);
print("Encoding categories test titles")
encoded_digits_titles_test = np_utils.to_categorical(randomized_digits_titles_test, 10);

# Reshape of arrays to have same neural dimm
classes = pow(2, 9)
pixels = 784

# Build the model
model = model(pixels, classes)

# Train model with 20 epochs that updates every 100 images and a verbose of 2 is used to format and reduce the output line
lel = model.fit(digits_train,
encoded_digits_titles_train,
batch_size=100,
epochs=20,
verbose=2,
validation_data=(digits_test, encoded_digits_titles_test))

score = model.evaluate(digits_test, encoded_digits_titles_test, verbose=0)
print("Error: {}".format(100 - score[1]*100))

# training the model and saving metrics in history
save_dir = "./results/"
model_name = './keras_mnist.h5'
model_path = os.path.join(model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

"""
# Display Images
plt.figure()

plt.suptitle('Test data', fontsize=10)
for i in range(rand_samples):
  #print(i)
  plt.subplot(4, rand_samples//2, i+1)
  plt.tight_layout()
  plt.imshow(randomized_digits_train[i], cmap='gray', interpolation='none')
  plt.title("{}".format(randomized_digits_titles_train[i]))
  plt.xticks([])
  plt.yticks([])

# show the plot
plt.show()
"""