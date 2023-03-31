import sys
import numpy as np
from random import randint
from matplotlib import pyplot

from MLTools.network.ConvLayer import ConvLayer as Conv
from MLTools.network.ConvLayer2 import ConvLayer
from MLTools.network.MaxPooling import MaxPooling
from MLTools.network.FlattenLayer import FlattenLayer
from MLTools.network.Layer import Layer
from MLTools.network.Network import Network
from MLTools.network import util
from MLTools.network import kernels

from keras.datasets import mnist
from keras.utils import np_utils

def modify_data(matrices):
    blur = Conv([kernels.gaussian_blur])
    n = matrices.shape[0]
    res = []

    for i in range(int(n/100)):
        modified = blur.forward_propagation(matrices[100*i:100*(i+1),:,:])
        padded = util.add_padding(modified, (int(28*1.5), int(28*1.5)))
        noise = np.random.randn(42, 42)*0.05*255
        final = np.clip(padded + noise, 0, 255)
        res.append(final)

    return np.concatenate(res)

if __name__ == "__main__":
    display = False
    if len(sys.argv) > 1 and sys.argv[1] == '-v':
        display = True 

    #loading the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("manipulating data")

    # same for test data : 10000 samples
    x_test = x_test.astype('float32')
    x_test = modify_data(x_test)
    x_test = x_test.reshape(x_test.shape[0], 1, 42, 42)
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
        Layer(8*4*9*9, 500, activation='tanh'),
        Layer(500, 200, activation='tanh'),
        Layer(200, 50, activation='tanh'),
        Layer(50, 10, activation='tanh')
    ])

    #train the network
    net.load_parameters("params/NoisyMnistParams")

    #making predictions
    n = 10
    indices = np.random.randint(0, len(x_test), n)
    predictions = net.predict(x_test[indices])

    #display predictions
    for j in range(0, n, 2):
        i1, i2 = indices[j], indices[j+1]
        expected, predicted = util.get_prediction(y_test[i1]), util.get_prediction(predictions[j])
        print("expected : " + str(y_test[i1]) + ", predicted : " + str(predictions[j]))
        expected2, predicted2 = util.get_prediction(y_test[i2]), util.get_prediction(predictions[j+1])
        print("expected : " + str(y_test[i2]) + ", predicted : " + str(predictions[j+1]))

        pyplot.subplot(330 + 1 + j%4*3)
        pyplot.imshow(x_test[i1][0], cmap=pyplot.get_cmap('gray'))
        pyplot.text(0, 60, "expected : " + str(expected) + ", predicted : " + str(predicted))

        pyplot.subplot(330 + 3 + j%4*3)
        pyplot.imshow(x_test[i2][0], cmap=pyplot.get_cmap('gray'))
        pyplot.text(0, 60, "expected : " + str(expected2) + ", predicted : " + str(predicted2))

        if display and j%4 == 0:
            pyplot.show()

    all_predictions = np.array(net.predict(x_test))
    accuracy = util.accuracy(all_predictions.reshape((len(y_test), 10)), y_test)
    print("accuracy : " + str(accuracy))