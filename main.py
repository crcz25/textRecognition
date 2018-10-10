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

# Download database from MNIST to train the model
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


#Get random samples from database to train
random_X_train = X_train[np.random.choice(X_train.shape[0], 4, replace=False), :]

#Get random samples from database to test
random_X_test = X_test[np.random.choice(X_test.shape[0], 4, replace=False), :]


plt.subplot(221)
plt.imshow(random_X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(random_X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(random_X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(random_X_train[3], cmap=plt.get_cmap('gray'))

# show the plot
plt.show()
