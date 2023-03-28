import unittest
import numpy as np

from network.Network import Network
from network.Layer import Layer
from network import util

class NetworkTest(unittest.TestCase):
    def testSetLayers(self):
        net = Network()
        
        self.assertEqual(0, len(net.layers))

        net.setLayers([
            Layer(2, 3),
            Layer(3, 1)
        ])

        self.assertEqual(2, len(net.layers))

    def testLogicalAND(self):
        net = Network()
        net.setLayers([
            Layer(2, 3, activation='tanh'),
            Layer(3, 1, activation='tanh')
        ])

        #creating the training data
        train = np.array([[[-1, -1]],[[-1, 1]],[[1, -1]],[[1, 1]]])
        finalResult = np.array([[[-1]], [[-1]], [[-1]], [[1]]])

        #training the network
        net.fit(train, finalResult, loss='mse', generation=1000, learning_rate=0.1, display=False)

        #making predictions
        predictions = net.predict(train)

        self.assertTrue(predictions[0][0] < 0)
        self.assertTrue(predictions[1][0] < 0)
        self.assertTrue(predictions[2][0] < 0)
        self.assertTrue(predictions[3][0] > 0)

    def testLogicalOR(self):
        net = Network()
        net.setLayers([
            Layer(2, 3, activation='tanh'),
            Layer(3, 1, activation='tanh')
        ])

        #creating the training data
        train = np.array([[[-1, -1]],[[-1, 1]],[[1, -1]],[[1, 1]]])
        finalResult = np.array([[[-1]], [[1]], [[1]], [[1]]])

        #training the network
        net.fit(train, finalResult, loss='mse', generation=1000, learning_rate=0.1, display=False)

        #making predictions
        predictions = net.predict(train)

        self.assertTrue(predictions[0][0] < 0)
        self.assertTrue(predictions[1][0] > 0)
        self.assertTrue(predictions[2][0] > 0)
        self.assertTrue(predictions[3][0] > 0)

