import numpy as np
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
    #loading the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # training data : 60000 samples
    # reshape and normalize input data
    print("manipulating data")
    x_train = x_train.astype('float32')
    x_train = modify_data(x_train)
    x_train = x_train.reshape(x_train.shape[0], 1, 42, 42)
    x_train = x_train/128 -0.244
    x_train = np.clip(x_train, -1, 1)

    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = np_utils.to_categorical(y_train)
    y_train = y_train * 2 - 1

    # same for test data : 10000 samples
    x_test = x_test.astype('float32')
    x_test = modify_data(x_test)
    x_test = x_test.reshape(x_test.shape[0], 1, 42, 42)
    x_test = x_test/128 -0.244
    x_test = np.clip(x_test, -1, 1)

    y_test = np_utils.to_categorical(y_test)
    y_test = y_test * 2 - 1

    print("start training")
    print(np.mean(x_train))
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
    net.fit(x_train, y_train, loss='mse', generation=35, learning_rate=0.075)

    #save parameters
    net.save_parameters("params/NoisyMnistParams")

    #making predictions
    n = 20
    predictions = net.predict(x_test[0:n])

    #display predictions
    for i in range(n):
        print("expected : " + str(y_test[i]) + ", predicted : " + str(predictions[i]))