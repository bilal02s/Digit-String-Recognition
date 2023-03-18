import numpy as np
from matplotlib import pyplot
from PIL import Image
from PIL import ImageOps
from sklearn.metrics import confusion_matrix

import util.util as util
import util.kernels as kernels
from network.Network import Network
from network.Layer import Layer
from network.ConvLayer import ConvLayer
from network.MaxPooling import MaxPooling
from network.FlattenLayer import FlattenLayer

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
    net.load_parameters("params/ParamsGeneric")

    #make predictions
    all_preds = []
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
        all_preds += list(np.argmax(all_predictions.reshape((len(y_test), 10)), axis=1).reshape(-1))
        accuracy = util.accuracy(all_predictions.reshape((len(y_test), 10)), y_test)
        print("digit " + str(digit) + ", accuracy : " + str(accuracy))

        n = 6
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

            if j%2 == 1:
               pyplot.show()

    y_true = np.array([i for i in range(10) for j in range(1000)])
    classes = [i for i in range(10)]
    conf_matrix = confusion_matrix(all_preds, y_true)
    print(conf_matrix)