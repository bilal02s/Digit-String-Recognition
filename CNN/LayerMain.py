import numpy as np
from matplotlib import pyplot

from MLTools.network.ConvLayer import ConvLayer
from MLTools.network.MaxPooling import MaxPooling
from MLTools.network.FlattenLayer import FlattenLayer
from MLTools.network import util
from MLTools.network import kernels

from keras.datasets import mnist
from keras.utils import np_utils

def modify_data(matrices):
    blur = ConvLayer([kernels.gaussian_blur])
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
    x_train = modify_data(x_train[0:200])
    x_train = x_train.reshape(x_train.shape[0], 1, 42, 42)

    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = np_utils.to_categorical(y_train)
    y_train = y_train * 2 - 1

    # same for test data : 10000 samples
    x_test = x_test.astype('float32')
    x_test = modify_data(x_test[0:200])
    x_test = x_test.reshape(x_test.shape[0], 1, 42, 42)

    y_test = np_utils.to_categorical(y_test)
    y_test = y_test * 2 - 1

    conv = ConvLayer([kernels.diagonal])
    maxpool = MaxPooling(kernels.one, 2)
    flatten = FlattenLayer()
    
    for i in range(0, 2): 
        pyplot.subplot(330 + 1)
        pyplot.imshow(x_train[i].squeeze(), cmap=pyplot.get_cmap('gray'))
        img = x_train[i]
        pyplot.subplot(330 + 1 +1)
        pyplot.imshow(img.squeeze(), cmap=pyplot.get_cmap('gray'))
        res = conv.forward_propagation(img)
        max = maxpool.forward_propagation(res)
        pyplot.subplot(330 + 1 +2)
        pyplot.imshow(res.squeeze(), cmap=pyplot.get_cmap('gray'))
        pyplot.subplot(330 + 1 +3)
        pyplot.imshow(max.squeeze(), cmap=pyplot.get_cmap('gray'))
        res = flatten.forward_propagation(max)
        pyplot.subplot(330 + 1 +4)
        pyplot.imshow(res.reshape((21, 21)), cmap=pyplot.get_cmap('gray'))
        pyplot.show() 