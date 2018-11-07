from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dropout, Dropout, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import urllib
import random
import sys
import argparse
import math

# First layer in the Sequential Model
def model(num_pixels, num_classes):
	model = Sequential()

	model.add(Dense(num_classes, activation='relu', input_shape=(num_pixels,)))
	model.add(Dropout(0.2))

	model.add(Dense(num_classes, activation='relu'))
	model.add(Dropout(0.2))

	model.add(Dense(num_classes, activation='softmax'))

	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model
