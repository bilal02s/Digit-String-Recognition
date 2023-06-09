import numpy as np
from MLTools.network.Layer import Layer
from MLTools.network.Network import Network
from MLTools.network import util

from keras.datasets import mnist
from keras.utils import np_utils

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # training data : 60000 samples
    # reshape and normalize input data
    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_train = x_train/128 #-0.13#128
    #x_train = np.clip(x_train, 0, 1)
    x_train = x_train -0.244
    x_train = np.clip(x_train, -1, 1)
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = np_utils.to_categorical(y_train)
    y_train = y_train*2 -1

    # same for test data : 10000 samples
    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    x_test = x_test/128 #-0.13#128
    #x_test = np.clip(x_test, 0, 1)
    x_test -= 0.244
    x_test = np.clip(x_test, -1, 1)
    y_test = np_utils.to_categorical(y_test)
    #y_test = y_test*2 -1

    #create the network
    print(np.mean(x_train))
    net = Network()

    #adding two layers
    net.setLayers([
        Layer(28*28, 250, activation='tanh'),
        Layer(250, 100, activation='tanh'),
        Layer(100, 20, activation='tanh'),
        Layer(20, 10, activation='tanh')
    ])

    #training the network
    net.fit(x_train, y_train, loss='mse', generation=100, learning_rate=0.025)

    #making predictions
    n = 20
    predictions = net.predict(x_test[0:n])

    #display predictions
    for i in range(n):
        print("expected : " + str(y_test[i]) + ", predicted : " + str(predictions[i]))
    
    #save parameters
    net.save_parameters("params/MnistParams")

