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
    blur = ConvLayer([kernels.gaussian])
    modified = []

    for matrix in matrices:
        [blurred] = blur.forward_propagation([matrix])
        modified.append(util.add_padding(blurred, (int(blurred.shape[0]*1.5), int(blurred.shape[1]*1.5))))

    return modified

if __name__ == "__main__":
    #loading the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # training data : 60000 samples
    # reshape and normalize input data
    #x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    print("manipulating data")
    x_train = x_train.astype('float32')
    x_train = x_train/128 - 1.0
    x_train = modify_data(x_train)
    x_train = [[sample] for sample in x_train]

    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = np_utils.to_categorical(y_train)
    y_train = y_train * 2 - 1

    # same for test data : 10000 samples
    #x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    x_test = x_test/128 - 1.0
    x_test = modify_data(x_test)
    x_test = [[sample] for sample in x_test]

    y_test = np_utils.to_categorical(y_test)
    y_test = y_test * 2 - 1

    print("start training")
    #create the network
    net = Network()

    #setting the error and activation functions
    net.setErrorFunction(util.mse, util.mse_prime)
    net.setActivationFunction(util.tanh, util.tanh_prime)

    #add layers
    net.addLayer(ConvLayer([kernels.horizontal, kernels.vertical, kernels.diagonal]))
    net.addLayer(MaxPooling(kernels.one, stride=2))
    net.addLayer(FlattenLayer())
    net.addLayer(Layer(3*21*21, 16))
    net.addLayer(Layer(16, 16))
    net.addLayer(Layer(16, 10))

    #train the network
    net.fit(x_train, y_train, generation=35, learning_rate=0.1, printOn=1)

    #making predictions
    n = 10
    predictions = net.predict(x_test[0:n])

    #display predictions
    for i in range(n):
        print("expected : " + str(y_test[i]) + ", predicted : " + str(predictions[i]))
    
    #save parameters
    net.save_parameters("params/MnistParams2.txt")



     