import unittest 
import numpy as np

from network.Layer import Layer
from network import util

class LayerTest(unittest.TestCase):
    def setUp(self):
        self.layer = Layer(10, 5, activation='tanh')
        self.layer.weights = np.tile(np.array([0, 0, 1, 0, 0]), 10).reshape(10, 5).astype('float64')
        self.layer.bias = np.array([1, 1, 0, 1, 1]).reshape(1, 5).astype('float64')

    def testGetters(self):
        layer = Layer(10, 5, activation='sigmoid')
        self.assertEqual(layer.getInputSize(), 10)
        self.assertEqual(layer.getOutputSize(), 5)
        self.assertEqual(layer.activation, util.sigmoid)
        self.assertEqual(layer.activation_prime, util.sigmoid_prime)

    def testForwardPropagation(self):
        input = np.arange(10)/10
        expected_output = util.tanh(np.array([1, 1, input.sum(), 1, 1]))
        computed_output = self.layer.forward_propagation(input)

        comp = expected_output == computed_output
        self.assertTrue(comp.all())

    def testBackpropagation(self):
        def identity(x):
            return x 

        def one(x):
            return 1

        init_bias = self.layer.bias.copy()
        init_weights = self.layer.weights.copy()
        
        self.layer.activation = identity 
        self.layer.activation_prime = one

        input = np.arange(1, 11).reshape(1, 10)/10
        output = np.array([1, 1, input.sum(), 1, 1])
        error = np.array([1., 2., 3., 2., 1.]).reshape(1, 5)

        self.layer.forward_propagation(input)
        input_err = self.layer.backpropagation(error, learning_rate=0.1)

        expected_bias = init_bias - error*0.1
        expected_weights = init_weights - input.T*error*0.1
        expected_in_err = np.tile(3., 10)

        comp1 = expected_bias == self.layer.bias 
        comp2 = expected_weights == self.layer.weights 
        comp3 = expected_in_err == input_err

        self.assertTrue(comp1.all())
        self.assertTrue(comp2.all())
        self.assertTrue(comp3.all())
