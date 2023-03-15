import numpy as np
from matplotlib import pyplot

import util.util as util
import util.kernels as kernels
from network.Network import Network
from network.Layer import Layer
from network.ConvLayer import ConvLayer
from network.MaxPooling import MaxPooling
from network.FlattenLayer import FlattenLayer

from keras.datasets import mnist
from keras.utils import np_utils

def modify_data(matrices):
    modified = []

    for matrix in matrices:
        noise = np.random.randn(28, 28)*0*255
        final = np.clip(matrix + noise, 0, 255)
        modified.append(final)

    return modified

if __name__ == "__main__":
    #loading the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # training data : 60000 samples
    # reshape and normalize input data
    print("manipulating data")
    x_train = x_train.astype('float32')
    train_X = modify_data(x_train[0:100])

    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = np_utils.to_categorical(y_train)
    y_train = y_train * 2 - 1

    # same for test data : 10000 samples
    x_test = x_test.astype('float32')
    test_X = modify_data(x_test[0:100])

    y_test = np_utils.to_categorical(y_test)
    y_test = y_test * 2 - 1
    
    digit = 6
    count = 0
    max = 10
    for i in range(0, 100):
        if np.argmax(y_train[i]) == digit:
            count += 1
            pyplot.subplot(330 + 1)
            pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
            pyplot.show() 
            print(x_train[i].mean())

        if count >= max:
            break