import unittest 
import numpy as np

from network.MaxPooling import MaxPooling
from network import util

class MaxPoolingTest(unittest.TestCase):
    def testFPpair(self):
        ''' testing correct behaviour when stride is set to 2 and image dimensions are pair '''
        kernel = np.ones((2, 2))
        layer = MaxPooling(kernel, stride=2)
        input = np.arange(6*6).reshape(1, 6, 6)
        expected_output = np.array([7., 9., 11., 19., 21., 23., 31., 33., 35.]).reshape(1, 3, 3)

        output = layer.forward_propagation(input)

        comp = expected_output == output 

        self.assertTrue(comp.all())

    def testFPunpair(self):
        ''' testing correct behaviour when stride is set to 2 and image dimensions are unpair '''
        kernel = np.ones((2, 2))
        layer = MaxPooling(kernel, stride=2)
        input = np.arange(7*7).reshape(1, 7, 7)
        expected_output = np.array([8., 10., 12., 22., 24., 26., 36., 38., 40.]).reshape(1, 3, 3)

        output = layer.forward_propagation(input)

        comp = expected_output == output 

        self.assertTrue(comp.all())

    def testFPstride3(self):
        ''' testing correct behaviour when stride is set to 3 and image dimensions are pair '''
        kernel = np.ones((2, 2))
        layer = MaxPooling(kernel, stride=3)
        input = np.arange(9*9).reshape(1, 9, 9)
        expected_output = np.array([10., 13., 16., 37., 40., 43., 64., 67., 70.]).reshape(1, 3, 3)

        output = layer.forward_propagation(input)

        comp = expected_output == output 

        self.assertTrue(comp.all())

    def testFPSeveralMatrices(self):
        kernel = np.ones((2, 2))
        layer = MaxPooling(kernel, stride=2)
        input = np.arange(64).reshape(4, 4, 4)
        expected_output = np.array([5, 7, 13, 15, 21, 23, 29, 31, 37, 39, 45, 47, 53, 55, 61, 63]).reshape(4, 2, 2).astype('float64')

        output = layer.forward_propagation(input)

        comp = expected_output == output 

        self.assertTrue(comp.all())

    def testBackProp(self):
        kernel = np.ones((2, 2))
        layer = MaxPooling(kernel, stride=2)
        input = np.arange(64).reshape(4, 4, 4)
        error = np.repeat(-1, 16).reshape(4, 2, 2)
        expected_in_err = np.tile([0, 0, 0, 0, 0, -1, 0, -1], 8).reshape(4, 4, 4)

        layer.forward_propagation(input)
        input_error = layer.backpropagation(error, learning_rate=None)

        comp = expected_in_err == input_error 

        self.assertTrue(comp.all())
