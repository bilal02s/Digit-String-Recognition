import numpy as np
from Layer import Layer
from Network import Network
import util 

from keras.datasets import mnist
from keras.utils import np_utils

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # training data : 60000 samples
    # reshape and normalize input data
    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_train /= 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = np_utils.to_categorical(y_train)

    # same for test data : 10000 samples
    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = np_utils.to_categorical(y_test)

    #create the network
    net = Network()

    #setting the error and activation functions
    net.setErrorFunction(util.mse, util.mse_prime)
    net.setActivationFunction(util.tanh, util.tanh_prime)

    #adding two layers
    net.addLayer(Layer(28*28, 100))
    net.addLayer(Layer(100, 50))
    net.addLayer(Layer(50, 10))

    #training the network
    net.fit(x_train, y_train, generation=35, learning_rate=0.1)

    #making predictions
    predictions = net.predict(x_test[0:3])

    #display predictions
    for i in range(3):
        print("expected : " + str(y_test[i]) + ", predicted : " + str(predictions[i]))
        

