import unittest 
import numpy as np

from network.FlattenLayer import FlattenLayer
from network import util

class FlattenLayerTest(unittest.TestCase):
    def testForwardPorpagation(self):
        layer = FlattenLayer()
        input = np.random.randn(4, 9)
        
        output = layer.forward_propagation(input)

        #test if the output is the input matrix flattened
        comp = input.reshape(1, -1) == output 

        self.assertTrue(comp.all())

    def testBackPropagation(self):
        layer = FlattenLayer()
        input = np.random.randn(3, 4, 9)
        error = np.random.randn(3*4*9)
        
        layer.forward_propagation(input)
        input_error = layer.backpropagation(error, learning_rate=None)

        # test if the input_error is the same error but reshaped to the input shape
        comp = error.reshape(3, 4, 9) == input_error

        self.assertTrue(comp.all())