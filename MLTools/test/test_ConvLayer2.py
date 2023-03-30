import unittest 
import numpy as np

from scipy.signal import correlate

from network.ConvLayer2 import ConvLayer
from network import kernels
from network import util

class ConvLayer2Test(unittest.TestCase):
    def testFP1(self):
        ''' test forward propagation using two predefined kernels, and setting the padding as "same" '''
        def identity(x):
            return x 

        layer = ConvLayer((2, 3, 3), activation='tanh', padding='same')
        image1 = np.diag([1, 2, 3, 4])*3
        image2 = np.ones((4, 4))*3
        images = np.concatenate([image1, image2]).reshape(2, 4, 4)

        #apply convolution operation using the scipy.signal module
        res = []
        for image in [image1, image2]:
            for kernel in [kernels.vertical, kernels.horizontal]:
                res.append(correlate(image, kernel, mode='same', method='direct'))
        expected_output = np.concatenate(res).reshape(4, 4, 4)

        #set Activation function
        layer.activation = identity

        #set kernels manually
        layer.kernels = np.array([kernels.vertical, kernels.horizontal])
        layer.bias = np.array([0., 0.])

        #perform forward propagation
        output = layer.forward_propagation(images)
        
        comp = expected_output == output 

        self.assertTrue(comp.all())

    def testFP2(self):
        ''' test forward propagation using two predefined kernels, and setting the padding as "valid" '''
        def identity(x):
            return x 

        layer = ConvLayer((2, 3, 3), activation='tanh', padding='valid')
        image1 = np.diag([1, 2, 3, 4])*3
        image2 = np.ones((4, 4))*3
        images = np.concatenate([image1, image2]).reshape(2, 4, 4)

        #apply convolution operation using the scipy.signal module
        res = []
        for image in [image1, image2]:
            for kernel in [kernels.vertical, kernels.horizontal]:
                res.append(correlate(image, kernel, mode='valid', method='direct'))
        expected_output = np.concatenate(res).reshape(4, 2, 2)

        #set Activation function
        layer.activation = identity

        #set kernels manually
        layer.kernels = np.array([kernels.vertical, kernels.horizontal])
        layer.bias = np.array([0., 0.])

        #perform forward propagation
        output = layer.forward_propagation(images)
        
        #compare results
        comp = expected_output == output 

        self.assertTrue(comp.all())

    def testBackProp1(self):
        ''' test backpropagation using two predefined kernels, and setting the padding as "same" '''
        def identity(x):
            return x 

        def one(x):
            return 1

        layer = ConvLayer((2, 3, 3), activation='tanh', padding='same')
        image1 = np.diag([1, 2, 3, 4])
        image2 = np.ones((4, 4))
        images = np.concatenate([image1, image2]).reshape(2, 4, 4)

        #apply convolution operation using the scipy.signal module
        res = []
        kers = np.array([kernels.vertical*3, kernels.horizontal*3])
        for image in images:
            for kernel in kers:
                res.append(correlate(image, kernel, mode='same', method='direct'))
        expected_output = np.concatenate(res).reshape(4, 4, 4)

        #set Activation function
        layer.activation = identity
        layer.activation_prime = one

        #set kernels manually
        layer.kernels = kers.copy()
        layer.bias = np.array([0., 0.])

        #defining error
        error = np.random.randint(0, 10, size=(4, 4, 4))

        #perform forward propagation
        output = layer.forward_propagation(images)
        input_err = layer.backpropagation(error.copy(), learning_rate=0.1)

        #finding expected values
        kers_err = [[], []]
        expected_in_err = [[], []]
        index=0
        for j in range(len(images)):
            for i in range(len(kers)):
                pad_img = np.pad(images[j], 1, mode='constant', constant_values=0)
                kers_err[i].append(correlate(pad_img, error[index], mode='valid', method='direct'))
                pad_err = np.pad(error[index], 1, mode='constant', constant_values=0)
                expected_in_err[j].append(correlate(pad_err, np.flip(kers[i]), mode='valid', method='direct'))
                index += 1

        kers_err = np.array(kers_err).sum(axis=1)
        expected_in_err = np.array(expected_in_err).sum(axis=1)
        
        expected_bias = np.array([0., 0.]) - 0.1*error.sum(axis=(1, 2)).reshape(2, 2).sum(axis=0)
        expected_kernels = kers - 0.1*kers_err
        
        #compare results
        comp1 = expected_bias == layer.bias 
        comp2 = expected_kernels == layer.kernels
        comp3 = expected_in_err == input_err

        self.assertTrue(comp1.all())
        self.assertTrue(comp2.all())
        self.assertTrue(comp3.all())

    def testBackProp2(self):
        ''' test backpropagation using two predefined kernels, and setting the padding as "valid" '''
        def identity(x):
            return x 

        def one(x):
            return 1

        layer = ConvLayer((2, 3, 3), activation='tanh', padding='valid')
        image1 = np.diag([1, 2, 3, 4])
        image2 = np.ones((4, 4))
        images = np.concatenate([image1, image2]).reshape(2, 4, 4)

        #apply convolution operation using the scipy.signal module
        res = []
        kers = np.array([kernels.vertical*3, kernels.horizontal*3])
        for image in images:
            for kernel in kers:
                res.append(correlate(image, kernel, mode='valid', method='direct'))
        expected_output = np.concatenate(res).reshape(4, 2, 2)

        #set Activation function
        layer.activation = identity
        layer.activation_prime = one

        #set kernels manually
        layer.kernels = kers.copy()
        layer.bias = np.array([0., 0.])

        #defining error
        error = np.random.randint(0, 10, size=(4, 2, 2))

        #perform forward propagation
        output = layer.forward_propagation(images)
        input_err = layer.backpropagation(error.copy(), learning_rate=0.1)

        #finding expected values
        kers_err = [[], []]
        expected_in_err = [[], []]
        index=0
        for j in range(len(images)):
            for i in range(len(kers)):
                kers_err[i].append(correlate(images[j], error[index], mode='valid', method='direct'))
                pad_err = np.pad(error[index], 2, mode='constant', constant_values=0)
                expected_in_err[j].append(correlate(pad_err, np.flip(kers[i]), mode='valid', method='direct'))
                index += 1

        kers_err = np.array(kers_err).sum(axis=1)
        expected_in_err = np.array(expected_in_err).sum(axis=1)
        
        expected_bias = np.array([0., 0.]) - 0.1*error.sum(axis=(1, 2)).reshape(2, 2).sum(axis=0)
        expected_kernels = kers - 0.1*kers_err
        
        #compare results
        comp1 = expected_bias == layer.bias 
        comp2 = expected_kernels == layer.kernels
        comp3 = expected_in_err == input_err

        self.assertTrue(comp1.all())
        self.assertTrue(comp2.all())
        self.assertTrue(comp3.all())