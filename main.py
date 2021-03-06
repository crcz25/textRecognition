from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model
from pprint import pprint

import numpy as np

import sys
import os
import argparse
import math

from network import model, webapp_model

from keras import backend as K
import tensorflowjs
import os

web = False

# Parameters
classes = 10
pixels = 784
dropout = 0.2

batch_size = 200
epochs = 20
verbose = 2


# Download database from MNIST to train the model
(digits_train, digits_titles_train), (digits_test, digits_titles_test) = mnist.load_data()


# We resize the array since mnist is structured as a 3-dimensional array, for the perceptron model we need to reduce the images to a one size vector
# an image of 28x28 is tranformed to 784 px
# Also we need to scale the pizels from 0-255 to 0-1 by dividing them by 255

if web:
  # For web app model
  print("Data for web model")
  digits_train = digits_train.reshape(digits_train.shape[0], 28, 28, 1).astype('float32')
  digits_test = digits_test.reshape(digits_test.shape[0], 28, 28, 1).astype('float32')
  digits_train /= 255
  digits_test /= 255
else :
  print("Data for CNN model")
  # Flatten and normalize pixel images to 0 and 1 on train dataset
  digits_train = digits_train.reshape(digits_train.shape[0], pixels).astype('float32')
  digits_test = digits_test.reshape(digits_test.shape[0], pixels).astype('float32')

  # Flatten and normalize pixel images to 0 and 1 on test dataset
  digits_train /= 255
  digits_test /= 255

  print("Train shape", digits_train[0].shape)

# Categorize the title corresponding to each digit using one-hot encoding from 0 to 9, transforming the vector of class integers into a binary matrix
# For example, the nmber 4 should be represented as:
#   [0,0,0,1,0,0,0,0,0,0]
encoded_titles_train = np_utils.to_categorical(digits_titles_train, classes)
print("Encoding categories train titles",  digits_titles_train.shape)

encoded_titles_test = np_utils.to_categorical(digits_titles_test, classes)
print("Encoding categories test titles",  digits_titles_test.shape)


# Divide the training set into:
#   Training Set (50,000)
#   Cross-validation (10,000)
#   Test Set (10,000)

#Divide data
digits_train = np.split(digits_train, [50000])
encoded_titles_train = np.split(encoded_titles_train, [50000])

aux_train =  np.copy(digits_train)
aux_titles_train =  np.copy(encoded_titles_train)

digits_train = aux_train[0]
digits_train_cross = aux_train[1]

encoded_titles_train = aux_titles_train[0]
encoded_titles_train_cross = aux_titles_train[1]


# print the final input shape ready for training
# print("Test on ", len(digits_test))
# print("Test shape", digits_test.shape)

if web:
  print("Training web model")
  # Web app model
  model = webapp_model()
  model_train = model.fit(digits_train, encoded_titles_train,
  validation_data=(digits_train_cross, encoded_titles_train_cross),
  epochs=epochs, batch_size=batch_size, verbose=verbose)

  # Final evaluation of the model
  scores = model.evaluate(digits_test, encoded_titles_test, verbose=verbose)
  print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

else:
  print("Training CNN model")
  # Build the model
  model = model(pixels, classes, dropout)

  # Train model with 20 epochs that updates every 200 images and a verbose of 2 is used to format and reduce the output line
  model_train = model.fit(digits_train, encoded_titles_train,
  validation_data=(digits_train_cross, encoded_titles_train_cross),
  batch_size=batch_size, epochs=epochs, verbose=verbose)


print("\nSaving model for webapp")
model_save_path = "output"
tensorflowjs.converters.save_keras_model(model, model_save_path)

# training the model and saving metrics in history
print("\nSaving model to calculate performance")
model_name = './keras_mnist.h5'
model_path = os.path.join(model_name)
model.save(model_path)

trained_model = load_model('./keras_mnist.h5')
print("\nCalculate performance")
performance = model.evaluate(digits_test, encoded_titles_test, verbose=verbose)

print("Create predictions based on tests")
predictions = trained_model.predict_classes(digits_test)

correct_indices = np.nonzero(predictions == digits_titles_test)[0]
incorrect_indices = np.nonzero(predictions != digits_titles_test)[0]
print("\nSuccesses:", len(correct_indices))
print("Errors:", len(incorrect_indices))

#Performance measures
print("Large CNN Error: %.2f%%" % (100-performance[1]*100))
print("\nLoss: {0:.4f}".format(performance[0]))
print("Accuracy: {0:.4f}".format(performance[1]))
print("Error: {0:.4f}\n".format(1 - performance[1]))

#print()
#pprint(model_train.__dict__)
#print(model_train.history.keys())

# Plott performance and graph the learning curve
stats = plt.figure()
#plt.suptitle('Stats', fontsize=16)

stats.add_subplot(2,1,1)
plt.plot(model_train.history['acc'])
plt.plot(model_train.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Test'], loc='lower right')

stats.add_subplot(2,1,2)
plt.plot(model_train.history['loss'])
plt.plot(model_train.history['val_loss'])
plt.title('Loss function')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.tight_layout()

# Show correct predictions
correct_predictions = plt.figure(figsize=(10, 10))
plt.suptitle('Correct samples predictions', fontsize=16)

for i, correct in enumerate(correct_indices[:15]):
  correct_predictions.add_subplot(3, 5, i + 1)
  plt.imshow(digits_test[correct].reshape(28,28), cmap='gray', interpolation='none')
  plt.title("Predicted: {}, Real: {}".format(predictions[correct], digits_titles_test[correct]))
  plt.xticks([])
  plt.yticks([])

plt.tight_layout()


# Show incorrect predictions
incorrect_predictions = plt.figure(figsize=(10, 10))
incorrect_predictions.subplots_adjust(hspace=0.4, wspace=0.4)
incorrect_predictions.suptitle('Incorrect sample predictions', fontsize=16)

for i, incorrect in enumerate(incorrect_indices[:15]):
  incorrect_predictions.add_subplot(3, 5, i + 1)
  plt.imshow(digits_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
  plt.title("Predicted {}, Real: {}".format(predictions[incorrect], digits_titles_test[incorrect]))
  plt.xticks([])
  plt.yticks([])

plt.tight_layout()

plt.show()


"""
from PIL import Image

trained_model = load_model('./keras_mnist.h5')
img = np.array(Image.open('./download.jpg').convert("L"))
img = img.reshape(1, pixels).astype('float32')
img /= 255
lel = np.array([0,0,1,0,0,0,0,0,0,0]).reshape(1, 10)
print(img.shape)
print(lel.shape)
performance = trained_model.evaluate(img, lel, verbose=verbose)
predictions = trained_model.predict_classes(img)
correct_indices = np.nonzero(predictions == lel)[0]
incorrect_indices = np.nonzero(predictions != lel)[0]
print("\nSuccesses:", len(correct_indices))
print("Errors:", len(incorrect_indices))

#Performance measures
print("Large CNN Error: %.2f%%" % (100-performance[1]*100))
print("\nLoss: {0:.4f}".format(performance[0]))
print("Accuracy: {0:.4f}".format(performance[1]))
print("Error: {0:.4f}\n".format(1 - performance[1]))
"""