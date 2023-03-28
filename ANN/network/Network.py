import numpy as np
from . import util

class Network:
    """
    A class representing a network that contains a list layers, forming a network of fully connected layers.

    ...

    Attributes
    ----------
    layers : list
        a list containing objects of class Layer

    Methods
    -------
    getInputSize():
        Returns the size of this input layer

    getOutputSize():
        returns the size of this output layer

    forward_propagation(input):
        performs a linear transformation of the input following by an activation function, returns an output array.

    backpropagation(error, learning_rate):
        performs backpropagation by updating the parameters using an error matrix and learning_rate value

    save_parameters(file):
        save this layer's weights and bias into the open binary file given in parameter

    load_parameters(file):
        load parameters into this layer's weights and bias from binary file given in parameter
    """
    def __init__(self):
        """
        Initialise layers attribute with an empty list, layers will be appended to this list

        """
        self.layers = []
        
    def addLayer(self, layer):
        """
        Add a new layer to the network
        the ordering of layers is import, each layer's input size needs to be equal to the output size of
        its previous layer, and each layer's output size needs to be equal to the input size of its next layer.

        Parameters
        ----------
            layer : Layer
                the layer to be added to the network
        """
        self.layers.append(layer)
    
    def setLayers(self, layers):
        """
        Set layers to the network
        the ordering of layers is import, each layer's input size needs to be equal to the output size of
        its previous layer, and each layer's output size needs to be equal to the input size of its next layer.

        Parameters
        ----------
            layers : list
                the layers to be setted to the network
        """
        self.layers = layers

    def fit(self, train, result, loss, generation, learning_rate, display=True):
        """
        Train the network on the given training set for a given number of epochs at a determined learning rate.

        Parameters
        ----------
            train : list
                list containing samples data
            result : list 
                list containing true output values
            generation : int
                the number of epochs of training on the dataset
            learning_rate : int
                the rate of learning
        """
        n = len(train)
        error_value = 0

        f, f_prime = util.get_loss_function(loss)

        for gen in range(generation):
            for i in range(n):
                output = train[i]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                error_value += f(output, result[i])
                error = f_prime(output, result[i])

                for layer in reversed(self.layers):
                    error = layer.backpropagation(error, learning_rate/(gen+1))

            error_value /= n

            if display:
                print("gen : " + str(gen) + ", error : " + str(error_value))

    def predict(self, toPredict):
        """
        Perform prediction on sample data given in parameters, returns list containing predictions

        Parameters
        ----------
            toPredict : list
                list containing sample data to make predictions on

        Return
        ------
            Returns a list containing the predictions to the given data samples
        """
        result = []

        for data in toPredict:
            output = data
            for layer in self.layers:
                output = layer.forward_propagation(output)

            result.append(output)

        return result

    def save_parameters(self, filename):
        """
        Opens a file whose path is specified in first parameter.
        saves the network learnable parameters into the file in binary format.

        Parameters
        ----------
            filename : str
                path to file where to store parameters
        """
        file = open(filename, 'wb')
        
        for layer in self.layers:
            layer.save_parameters(file)

        file.close()

    def load_parameters(self, filename):
        """
        Opens a file whose path is specified in first parameter.
        loads the network learnable parameters from the file.

        Parameters
        ----------
            filename : str
                path to file where to load parameters
        """
        file = open(filename, 'rb')

        for layer in self.layers:
            layer.load_parameters(file)
        
        file.close()

        