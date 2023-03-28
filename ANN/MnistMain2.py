import numpy as np
from network.Layer import Layer
from network.Network import Network
import network.util as util

from keras.datasets import mnist
from keras.utils import np_utils 

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # training data : 60000 samples
    # reshape and normalize input data
    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_train /= 128
    x_train = x_train -0.244
    x_train = np.clip(x_train, -1, 1)
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = np_utils.to_categorical(y_train)
    y_train *= 2
    y_train -= 1

    # same for test data : 10000 samples
    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    x_test /= 128
    x_train = x_train -0.244
    x_train = np.clip(x_train, -1, 1)
    y_test = np_utils.to_categorical(y_test)
    y_test *= 2
    y_test -= 1

    #create the network
    net = Network()

    #adding two layers
    net.setLayers([
        Layer(28*28, 250, activation='tanh'),
        Layer(250, 100, activation='tanh'),
        Layer(100, 20, activation='tanh'),
        Layer(20, 10, activation='tanh')
    ])

    #load trained parameters
    net.load_parameters("params/MnistParams3")

    #making predictions
    n = 20
    predictions = net.predict(x_test[0:n])

    #display predictions
    for i in range(n):
        print("expected : " + str(y_test[i]) + ", predicted : " + str(predictions[i]))
        print("expected : " + str((y_test[i]+1)/2) + ", predicted : " + str((predictions[i]+1)/2))
    
    all_predictions = np.array(net.predict(x_test))
    accuracy = util.accuracy(all_predictions.reshape((len(y_test), 10)), y_test)
    print("accuracy : " + str(accuracy))
