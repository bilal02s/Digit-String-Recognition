import numpy as np
from matplotlib import pyplot
from PIL import Image
from PIL import ImageOps
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
)
from sklearn.metrics import confusion_matrix

from MLTools.network.ConvLayer2 import ConvLayer
from MLTools.network.MaxPooling import MaxPooling
from MLTools.network.FlattenLayer import FlattenLayer
from MLTools.network.Layer import Layer
from MLTools.network.Network import Network
from MLTools.network import util
from MLTools.network import kernels

from TestPreprocessing import filterMatrix

if __name__ == "__main__":
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
    net.load_parameters("params/NewNetParams")

    #make predictions
    all_preds = []
    overall_accuracy = 0
    x = 6
    for digit in range(10):
        data = []
        y = [-1 for i in range(10)]
        y[digit] = 1
        y_test = np.array([y for i in range(1000)])
        for i in range(1, 1001): 
            index = i
            path = 'data/' + str(digit) + '/' + str(digit) + '_' + str(index) + '.jpg'
            image = np.array(ImageOps.invert(Image.open(path).convert('L').resize((28, 28)).filter(SHARPEN))).astype('float32')
            min, max = np.quantile(image, 0.25), image.max()
            image = np.dot(image, np.diag(np.repeat(255/(max-min), 28))) + np.repeat((255*min)/(min-max), 28).reshape((-1, 1))
            image = np.clip(image, 0, 255) 
            image = filterMatrix(image)
            data.append([image/128])
        data = np.array(data)
        data = (data -np.mean(data))/np.std(data)

        all_predictions = np.array(net.predict(data))
        all_preds += list(np.argmax(all_predictions.reshape((len(y_test), 10)), axis=1).reshape(-1))
        accuracy = util.accuracy(all_predictions.reshape((len(y_test), 10)), y_test)
        overall_accuracy += accuracy
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

    print("overall accuracy : " + str(overall_accuracy/10))
    y_true = np.array([i for i in range(10) for j in range(1000)])
    classes = [i for i in range(10)]
    conf_matrix = confusion_matrix(all_preds, y_true)
    print(conf_matrix)