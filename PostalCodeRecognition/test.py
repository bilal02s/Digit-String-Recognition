import numpy as np
from random import randint
from matplotlib import pyplot
from PIL import Image

from MLTools.network.ConvLayer2 import ConvLayer
from MLTools.network.MaxPooling import MaxPooling
from MLTools.network.FlattenLayer import FlattenLayer
from MLTools.network.Layer import Layer
from MLTools.network.Network import Network
from MLTools.network import util
from MLTools.network import kernels

from MatrixProcessing import MyMatrix

from keras.datasets import mnist
from keras.utils import np_utils

def modify_data(matrices):
    modified = []

    for matrix in matrices:
        matrix1 = MyMatrix(matrix)
        matrix2 = MyMatrix(matrix.copy())
        matrix1.addRandomNoise(10).randomTransformation((-10, 0), ((-2, -2), (2, 2)))
        r = randint(-6, 3)
        matrix2.zoom((r, r, 28-r, 28-r)).randomTransformation((-30, 15), ((-5, -5), (5, 5))).addRandomNoise(10).addRandomScratch(150).blur()

        modified.append(matrix1.getMatrix())
        modified.append(matrix2.getMatrix())

    return (np.array(modified)/128) -0.244

if __name__ == "__main__":
    #loading the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("manipulating data")

    # same for test data : 10000 samples
    x_test = x_test.astype('float32')
    x_test = modify_data(x_test)
    x_test = (x_test-np.mean(x_test))/np.std(x_test)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

    y_test = np_utils.to_categorical(y_test)
    y_test = y_test * 2 - 1
    y_test = np.repeat(y_test, 2, axis=0)

    print("making predictions")

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

    #load parameters
    net.load_parameters("params/NewNetParams")

    print(net.layers[0].kernels)
    print(net.layers[2].kernels)

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

        if j%2 == 1:
            pyplot.show()

    all_predictions = np.array(net.predict(x_test))
    accuracy = util.accuracy(all_predictions.reshape((len(y_test), 10)), y_test)
    print("accuracy : " + str(accuracy))
    



     
