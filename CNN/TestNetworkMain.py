import numpy as np
from random import randint
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
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("manipulating data")

    # same for test data : 10000 samples
    x_test = x_test.astype('float32')
    x_test = np.array([[sample] for sample in x_test])

    y_test = np_utils.to_categorical(y_test)
    y_test = y_test * 2 - 1

    print("start training")
    #create the network
    net = Network()

    #setting the error and activation functions
    net.setErrorFunction(util.mse, util.mse_prime)
    net.setActivationFunction(util.tanh, util.tanh_prime)

    #add layers
    net.addLayer(ConvLayer([kernels.horizontal, kernels.vertical, kernels.diagonal, kernels.diagonal2], (28, 28)))
    net.addLayer(MaxPooling(kernels.one, stride=2, input_shape=(28, 28)))
    net.addLayer(ConvLayer([kernels.horizontal, kernels.vertical, kernels.diagonal], (14, 14)))
    net.addLayer(MaxPooling(kernels.one, stride=2, input_shape=(14, 14)))
    net.addLayer(FlattenLayer())
    net.addLayer(Layer(12*7*7, 250))
    net.addLayer(Layer(250, 50))
    net.addLayer(Layer(50, 10))

    #train the network
    net.load_parameters("params/MnistParams3")

    #making predictions
    n = 10
    indices = np.random.randint(0, len(x_test), n)
    predictions = net.predict(x_test[indices])

    #display predictions
    for j in range(0, n):
        i = indices[j]
        expected, predicted = util.get_prediction(y_test[i]), util.get_prediction(predictions[j])
        print("expected : " + str(y_test[i]) + ", predicted : " + str(predictions[j]))

        pyplot.subplot(330 + 1 + j%2)
        pyplot.imshow(x_test[i][0], cmap=pyplot.get_cmap('gray'))
        pyplot.text(10*(j%2), 70, "expected : " + str(expected) + ", predicted : " + str(predicted))

        #if j%2 == 1:
         #   pyplot.show()

    all_predictions = np.array(net.predict(x_test))
    accuracy = util.accuracy(all_predictions.reshape((len(y_test), 10)), y_test)
    print("accuracy : " + str(accuracy))
    



     
