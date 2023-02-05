import numpy as np
from matplotlib import pyplot
import ANN.util
import kernels
from ANN.Network import Network
from ANN.Layer import Layer
from ConvLayer import ConvLayer
from MaxPooling import MaxPooling
from FlattenLayer import FlattenLayer

from keras.datasets import mnist
from keras.utils import np_utils

if __name__ == "__main__":
    #loading the dataset
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    #printing the shapes of the vectors 
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))

    conv = ConvLayer([kernels.vertical])
    maxpool = MaxPooling(kernels.one, 2)

    for i in range(0, 2):  
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
        [res] = conv.forward_propagation([train_X[i]])
        [max] = maxpool.forward_propagation([res])
        pyplot.subplot(330 + 1 + i+1)
        pyplot.imshow(res, cmap=pyplot.get_cmap('gray'))
        pyplot.subplot(330 + 1 + i+2)
        pyplot.imshow(max, cmap=pyplot.get_cmap('gray'))
        [res] = conv.forward_propagation([max])
        [max] = maxpool.forward_propagation([res])
        pyplot.subplot(330 + 1 + i+3)
        pyplot.imshow(res, cmap=pyplot.get_cmap('gray'))
        pyplot.subplot(330 + 1 + i+4)
        pyplot.imshow(max, cmap=pyplot.get_cmap('gray'))
        pyplot.show() 