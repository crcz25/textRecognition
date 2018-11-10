from keras.datasets import mnist
import numpy

(digits_train, digits_titles_train), (digits_test, digits_titles_test) = mnist.load_data()
aux = numpy.array_split(digits_train, 3)
lel = 12000
tot = 48000

print(len(numpy.split(digits_train, [48000, 54000])[2]))