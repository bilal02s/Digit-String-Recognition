import unittest

# import the test modules to include in the test suite
from test.test_Layer import LayerTest
from test.test_Network import NetworkTest
from test.test_FlattenLayer import FlattenLayerTest
from test.test_MaxPooling import MaxPoolingTest
from test.test_ConvLayer import ConvLayerTest
from test.test_ConvLayer2 import ConvLayer2Test

if __name__ == '__main__':
    # create a test suite
    test_suite = unittest.TestSuite()

    # add the test classes to the test suite
    test_suite.addTest(unittest.makeSuite(LayerTest))
    test_suite.addTest(unittest.makeSuite(NetworkTest))
    test_suite.addTest(unittest.makeSuite(FlattenLayerTest))
    test_suite.addTest(unittest.makeSuite(MaxPoolingTest))
    test_suite.addTest(unittest.makeSuite(ConvLayerTest))
    test_suite.addTest(unittest.makeSuite(ConvLayer2Test))

    # initialize a test runner
    runner = unittest.TextTestRunner()

    # run the test suite
    runner.run(test_suite)