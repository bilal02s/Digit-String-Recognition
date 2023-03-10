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
    blur = ConvLayer([kernels.gaussian_blur])
    modified = []

    for matrix in matrices:
        [blurred] = blur.forward_propagation([matrix])
        padded = util.add_padding(blurred, (int(blurred.shape[0]*1.5), int(blurred.shape[1]*1.5)))
        noise = np.random.randn(42, 42)*0.05*255
        final = np.clip(padded + noise, 0, 255)
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

    conv = ConvLayer([kernels.diagonal])
    maxpool = MaxPooling(kernels.one, 2)
    flatten = FlattenLayer()
    
    for i in range(0, 2): 
        pyplot.subplot(330 + 1)
        pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
        img = train_X[i]
        pyplot.subplot(330 + 1 +1)
        pyplot.imshow(img, cmap=pyplot.get_cmap('gray'))
        [res] = conv.forward_propagation([img])
        [max] = maxpool.forward_propagation([res])
        pyplot.subplot(330 + 1 +2)
        pyplot.imshow(res, cmap=pyplot.get_cmap('gray'))
        pyplot.subplot(330 + 1 +3)
        pyplot.imshow(max, cmap=pyplot.get_cmap('gray'))
        [res] = flatten.forward_propagation([max])
        pyplot.subplot(330 + 1 +4)
        pyplot.imshow(res.reshape((21, 21)), cmap=pyplot.get_cmap('gray'))
        pyplot.show() 