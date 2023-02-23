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

if __name__ == "__main__":
    #loading the dataset
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    #printing the shapes of the vectors 
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))

    conv = ConvLayer([kernels.gaussian])
    maxpool = MaxPooling(kernels.one, 2)

    for i in range(0, 2): 
        img = util.add_padding(train_X[i], (56, 56)) 
        pyplot.subplot(330 + 1)
        pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
        pyplot.subplot(330 + 2)
        pyplot.imshow(img, cmap=pyplot.get_cmap('gray'))
        [res] = conv.forward_propagation([img])
        [max] = maxpool.forward_propagation([res])
        pyplot.subplot(330 + 1 +2)
        pyplot.imshow(res, cmap=pyplot.get_cmap('gray'))
        pyplot.subplot(330 + 1 +3)
        pyplot.imshow(max, cmap=pyplot.get_cmap('gray'))
        [res] = conv.forward_propagation([max])
        [max] = maxpool.forward_propagation([res])
        pyplot.subplot(330 + 1 +4)
        pyplot.imshow(res, cmap=pyplot.get_cmap('gray'))
        pyplot.subplot(330 + 1 +5)
        pyplot.imshow(max, cmap=pyplot.get_cmap('gray'))
        pyplot.show() 