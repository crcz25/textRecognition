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

# First layer in the Sequential Model
def model(num_pixels, num_classes):
	model = Sequential()
	model.add(Dense(num_classes, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model
