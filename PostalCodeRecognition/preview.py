import numpy as np
from matplotlib import pyplot
from PIL import Image
from PIL import ImageOps
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
)
from sklearn.metrics import confusion_matrix

import util.util as util
import util.kernels as kernels
from network.Network import Network
from network.Layer import Layer
from network.ConvLayer import ConvLayer
from network.MaxPooling import MaxPooling
from network.FlattenLayer import FlattenLayer

from preprocessing import filterMatrix

if __name__ == "__main__":
    digit = 2
    for index in range(1, 1001):
        path = 'data/' + str(digit) + '/' + str(digit) + '_' + str(index) + '.jpg'
        image = np.array(ImageOps.invert(Image.open(path).convert('L').resize((28, 28)).filter(SHARPEN))).astype('float32')
        min, max = np.quantile(image, 0.25), image.max()
        image = np.dot(image, np.diag(np.repeat(255/(max-min), 28))) + np.repeat((255*min)/(min-max), 28).reshape((-1, 1))
        image = np.clip(image, 0, 255) 
        filtered = filterMatrix(image)

        pyplot.subplot(330 + 1 + 0)#index%2)
        pyplot.imshow(image, cmap=pyplot.get_cmap('gray'))
        pyplot.subplot(330 + 1 + 1)#index%2)
        pyplot.imshow(filtered, cmap=pyplot.get_cmap('gray'))

        #if index%2==1:
        #pyplot.show()