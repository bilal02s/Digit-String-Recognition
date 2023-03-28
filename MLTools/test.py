import unittest

# import the test modules to include in the test suite
from test.test_Layer import LayerTest
from test.test_Network import NetworkTest

if __name__ == '__main__':
    # create a test suite
    test_suite = unittest.TestSuite()

    # add the test classes to the test suite
    test_suite.addTest(unittest.makeSuite(LayerTest))
    test_suite.addTest(unittest.makeSuite(NetworkTest))

    # initialize a test runner
    runner = unittest.TextTestRunner()

    # run the test suite
    runner.run(test_suite)