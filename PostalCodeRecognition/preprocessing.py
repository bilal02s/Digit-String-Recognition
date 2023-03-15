import numpy as np
from matplotlib import pyplot
from PIL import Image
from PIL import ImageOps

import util.util as util
import util.kernels as kernels
from network.Network import Network
from network.Layer import Layer
from network.ConvLayer import ConvLayer
from network.MaxPooling import MaxPooling
from network.FlattenLayer import FlattenLayer

def modify_data(matrices):
    blur = ConvLayer([kernels.gaussian_blur])
    modified = []

    for matrix in matrices:
        [blurred] = blur.forward_propagation([matrix])
        padded = util.add_padding(blurred, (int(blurred.shape[0]*1.5), int(blurred.shape[1]*1.5)))
        noise = np.random.randn(42, 42)*0.1*255
        final = np.clip(padded + noise, 0, 255)
        modified.append(final)

    return modified

if __name__ == "__main__":
    #create the network
    net = Network()

    #setting the error and activation functions
    net.setErrorFunction(util.mse, util.mse_prime)
    net.setActivationFunction(util.tanh, util.tanh_prime)

    #add layers
    net.addLayer(ConvLayer([kernels.horizontal, kernels.vertical, kernels.diagonal, kernels.diagonal2]))
    net.addLayer(MaxPooling(kernels.one, stride=2))
    net.addLayer(ConvLayer([kernels.horizontal, kernels.vertical, kernels.diagonal]))
    net.addLayer(MaxPooling(kernels.one, stride=2))
    net.addLayer(FlattenLayer())
    net.addLayer(Layer(12*7*7, 250))
    net.addLayer(Layer(250, 50))
    net.addLayer(Layer(50, 10))

    #train the network
    net.load_parameters("params/MnistParams2conv")

    #make predictions
    x = 6
    for digit in range(10):
        data = []
        y = [-1 for i in range(10)]
        y[digit] = 1
        y_test = np.array([y for i in range(1000)])
        for i in range(1, 1001): 
            index = i
            path = 'data/' + str(digit) + '/' + str(digit) + '_' + str(index) + '.jpg'
            image = np.array(ImageOps.invert(Image.open(path).convert('L').resize((28, 28)))).astype('float32')
            min, max = np.quantile(image, 0.25), image.max()
            image = np.dot(image, np.diag(np.repeat(255/(max-min), 28))) + np.repeat((255*min)/(min-max), 28).reshape((-1, 1))
            image = np.clip(image, 0, 255)
            data.append([image])
        data = np.array(data)

        all_predictions = np.array(net.predict(data))
        accuracy = util.accuracy(all_predictions.reshape((len(y_test), 10)), y_test)
        print("digit " + str(digit) + ", accuracy : " + str(accuracy))

        n = 10
        indices = np.random.randint(0, len(data), n)
        predictions = net.predict(data[indices])

        #display predictions
        for j in range(0, n):
            i = indices[j]
            expected, predicted = util.get_prediction(y_test[i]), util.get_prediction(predictions[j])
            #print("expected : " + str(y_test[i]) + ", predicted : " + str(predictions[j]))

            #print(data[i][0].mean())
            pyplot.subplot(330 + 1 + j%2)
            pyplot.imshow(data[i][0], cmap=pyplot.get_cmap('gray'))
            pyplot.text(10*(j%2), 70, "expected : " + str(expected) + ", predicted : " + str(predicted))

            #if j%2 == 1:
             #  pyplot.show()