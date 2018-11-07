from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.models import load_model
from pprint import pprint

import numpy as np

import sys
import os
import argparse
import math

from network import model

# Download database from MNIST to train the model
(digits_train, digits_titles_train), (digits_test, digits_titles_test) = mnist.load_data()

# Flatten and normalize pixel images to 0 and 1 on train dataset
num_pixels = digits_train.shape[1] * digits_train.shape[2]
digits_train = digits_train.reshape(digits_train.shape[0], num_pixels).astype('float32')
digits_train /= 255

# Flatten and normalize pixel images to 0 and 1 on test dataset
num_pixels = digits_test.shape[1] * digits_test.shape[2]
digits_test = digits_test.reshape(digits_test.shape[0], num_pixels).astype('float32')
digits_test /= 255

# Categorize the title digit numbers using one-hot encoding from 0 to 9
# For example, the nmber 4 should be represented as:
#   [0,0,0,1,0,0,0,0,0,0]

# Encode categories from 0 to 9 using one-hot encoding
print("Encoding categories train titles")
digits_titles_train = np_utils.to_categorical(digits_titles_train);

print("Encoding categories test titles")
digits_titles_test = np_utils.to_categorical(digits_titles_test);

# Reshape of arrays to have same neural dimm
classes = 10
pixels = 784
print(classes)

# Build the model
model = model(pixels, classes)

# Train model with 20 epochs that updates every 200 images and a verbose of 2 is used to format and reduce the output line
model_train = model.fit(digits_train, digits_titles_train,
batch_size=250,
epochs=40,
verbose=2,
validation_data=(digits_test, digits_titles_test))

# training the model and saving metrics in history
save_dir = "./results/"
model_name = './keras_mnist.h5'
model_path = os.path.join(model_name)
model.save(model_path)
print('\nSaved trained model at %s ' % model_path)

model_stats = load_model('./keras_mnist.h5')
model_performance = model_stats.evaluate(digits_test, digits_titles_test, verbose=2)
print("\nLoss: {0:.4f}".format(model_performance[0]))
print("Accuracy: {0:.4f}".format(model_performance[1]))
print("Error: {0:.4f}".format(1-model_performance[1]))

# Plott performance and graph the learning curve
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(model_train.history['acc'])
plt.plot(model_train.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(model_train.history['loss'])
plt.plot(model_train.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

plt.show()

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