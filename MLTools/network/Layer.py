import numpy as np
from . import util

class Layer:
    """
    A class representing a layer in a fully connected neural network,
    implementing froward and backward propagation.

    ...

    Attributes
    ----------
    input_size : int
        number of input values
    output_size : int
        number of output values
    weights : numpy.ndarray
        array containing the weights whose shape is (nb_input, nb_output)
    bias : numpy.ndarray
        array containing the bias whose shape is (1, nb_output)
    activation : function
        the activation function
    activation_prime : function
        the derivative of the activation function
    input : numpy.ndarray
        array containing an input through a call to forward propagation (to be used to backpropagate)
    output : numpy.ndarray
        array containing the output of the corresponding input after the forward propagation (to be used to backpropagate)

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
    def __init__(self, input_size, output_size, activation='tanh'):
        """
        Initialise weights and bias matrices of the specified shapes with random values.
        set other attributes as necessary.

        Parameters
        ----------
            input_size : int
                size of the layer's input
            output_size : int
                size of the layer's output
            activation : str, optional
                the name of the activation function to use, (default is tanh)
        """
        self.weights = np.random.default_rng().normal(0, 1/np.sqrt(input_size), size=(input_size, output_size))
        self.bias = np.repeat(0, output_size).reshape(1, output_size).astype('float64')
        self.input_size = input_size
        self.output_size = output_size
        self.activation, self.activation_prime = util.get_activation_function(activation)

    def getInputSize(self):
        """
        Returns the size of this layer's input

        Returns
        -------
            The size of this layer's input
        """
        return self.input_size

    def getOutputSize(self):
        """
        Returns the size of this layer's output

        Returns
        -------
            The size of this layer's output
        """
        return self.output_size

    def forward_propagation(self, input):
        """
        Performs forward propagation, essentially a dot product between the input and the weights array.
        Apply the activation function to the result, and then returns the final result as output.

        Parameters
        ----------
        input : numpy.ndarray
            matrix of shape (1, input_size) or (input_size,)

        Returns
        -------
        output : numpy.ndarray
            matrix of shape (1, output_size) or (output_size), (same as input matrix)
        """
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        activationOutput = self.activation(self.output)

        return activationOutput

    def backpropagation(self, error, learning_rate):
        """
        Performs backpropagation, and updating the weights and bias accordingly
        calculates the error of the input used in the last forward propagation and returns it as an array
        of the same size as the input size

        Parameters
        ----------
        error : numpy.ndarray
            matrix of shape (1, output_size) or (output_size,)
        learning_rate : float
            the learning rate value

        Returns
        -------
        output : numpy.ndarray
            matrix of shape (1, input_size) or (input_shape,), (same as error matrix)
        """
        # finding (dE/dz) * f'(y)
        common = error * self.activation_prime(self.output)
        bias_err = common
        input_err = np.dot(common, self.weights.T)
        weights_err = np.dot(self.input.T, common)
        
        #adjusting the parameters
        self.weights -= learning_rate * weights_err
        self.bias -= learning_rate * bias_err

        #print(input_err)

        return input_err

    def save_parameters(self, file):
        """
        Saves this layer's parameters (weights and bias) into the open file given in parameter

        Parameters
        ----------
        file : open file
            an open file to write
        """
        self.weights.tofile(file)
        self.bias.tofile(file)

    def load_parameters(self, file):
        """
        Loads from the open file given in parameter, the matrices that make up this layer's parameters (weights and bias)

        Parameters
        ----------
        file : open file
            an open file to read
        """
        weightsCount = self.weights.size
        biasCount = self.bias.size

        self.weights = np.fromfile(file, count=weightsCount).reshape(self.weights.shape)
        self.bias = np.fromfile(file, count=biasCount).reshape(self.bias.shape)