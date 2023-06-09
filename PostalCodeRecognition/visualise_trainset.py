import numpy as np
from matplotlib import pyplot
from PIL import Image
from random import randint

from MLTools.network.ConvLayer2 import ConvLayer
from MLTools.network.MaxPooling import MaxPooling
from MLTools.network.FlattenLayer import FlattenLayer
from MLTools.network.Layer import Layer
from MLTools.network.Network import Network
from MLTools.network import util
from MLTools.network import kernels

from MatrixProcessing import MyMatrix

from keras.datasets import mnist
from keras.utils import np_utils

def modify_data(matrices):
    modified = []

    for matrix in matrices:
        matrix = np.pad(matrix, 7, mode='constant', constant_values=0)
        matrix1 = MyMatrix(matrix)
        matrix2 = MyMatrix(matrix.copy())
        matrix1.zoom((7, 7, 35, 35)).addRandomNoise(10).randomTransformation((-10, 0), ((-2, -2), (2, 2)))
        r = randint(1, 10)
        matrix2.zoom((r, r, 42-r, 42-r)).randomTransformation((-30, 15), ((-5, -5), (5, 5))).addRandomNoise(10).addRandomScratch(200).blur()

        modified.append(matrix1.getMatrix())
        modified.append(matrix2.getMatrix())

    return modified

if __name__ == "__main__":
    #loading the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # training data : 60000 samples
    # reshape and normalize input data
    print("manipulating data")
    x_train = x_train.astype('float32')
    x_train = modify_data(x_train[0:100])

    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = np_utils.to_categorical(y_train)
    y_train = y_train * 2 - 1
    y_train = np.repeat(y_train, 2, axis=0)

    # same for test data : 10000 samples
    x_test = x_test.astype('float32')
    x_test = modify_data(x_test[0:100])

    y_test = np_utils.to_categorical(y_test)
    y_test = y_test * 2 - 1
    y_test = np.repeat(y_test, 2, axis=0)
    
    digit = 6
    count = 0
    max = 10
    for i in range(0, 100):
        if np.argmax(y_train[i]) == digit or True:
            count += 1
            pyplot.subplot(330 + 1)
            pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
            pyplot.show() 

        if count >= max:
            break