import unittest 
import numpy as np

from scipy.signal import correlate

from network.ConvLayer import ConvLayer
from network import kernels
from network import util

class ConvLayerTest(unittest.TestCase):
    def testForwardPorpagation(self):
        layer = ConvLayer(np.array([kernels.vertical, kernels.horizontal]))
        image1 = np.diag([1, 2, 3, 4])*3
        image2 = np.ones((4, 4))*3
        images = np.concatenate([image1, image2]).reshape(2, 4, 4)

        #apply convolution operation using the scipy.signal module
        res = []
        for image in [image1, image2]:
            for kernel in [kernels.vertical, kernels.horizontal]:
                res.append(correlate(image, kernel, mode='same', method='direct'))
        expected_output = np.concatenate(res).reshape(4, 4, 4)

        output = layer.forward_propagation(images)
        
        comp = expected_output == output 

        self.assertTrue(comp.all())