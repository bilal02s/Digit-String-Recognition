import numpy as np
from random import randint
from matplotlib import pyplot

from MLTools.network.ConvLayer2 import ConvLayer
from MLTools.network.MaxPooling import MaxPooling
from MLTools.network.FlattenLayer import FlattenLayer
from MLTools.network.Layer import Layer
from MLTools.network.Network import Network
from MLTools.network import util
from MLTools.network import kernels

from keras.datasets import mnist
from keras.utils import np_utils

if __name__ == "__main__":
    #loading the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("manipulating data")

    # same for test data : 10000 samples
    x_test = x_test.astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    x_test = x_test/128 -0.244
    x_test = np.clip(x_test, -1, 1)

    y_test = np_utils.to_categorical(y_test)
    y_test = y_test * 2 - 1

    print("start training")
    #create the network
    net = Network()

    #add layers
    net.setLayers([
        ConvLayer((8, 3, 3), activation='tanh', padding='valid'),
        MaxPooling(kernels.one, stride=2),
        ConvLayer((4, 3, 3), activation='tanh', padding='valid'),
        MaxPooling(kernels.one, stride=2),
        FlattenLayer(),
        Layer(8*4*5*5, 250, activation='tanh'),
        Layer(250, 80, activation='tanh'),
        Layer(80, 10, activation='tanh')
    ])

    #train the network
    net.load_parameters("params/MnistParams")

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
    



     
