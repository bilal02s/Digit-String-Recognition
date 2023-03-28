import numpy as np
from random import randint

# Mean Squared Error and its derivative
def mse(prediction, expected):
    '''Computes and return the mean square error, between predicted vector and true value vector'''
    return np.mean(np.power(prediction - expected, 2))

def mse_prime(prediction, expected):
    '''Computes the derivative of the mean squared error with respect to a predicted vector and its true value'''
    return 2*(prediction - expected)/expected.size

# activation functions and their derivatives
# parabolic tangent
def tanh(x):
    '''Applies hyperbolic tangent to all elements of a given vector/matrix, returns the result'''
    return np.tanh(x);

def tanh_prime(x):
    '''Applies the derivative of hyperbolic tangent to all elements of a given vector/matrix, returns the result'''
    return 1-np.tanh(x)**2;

def sigmoid(x):
    '''Applies sigmoid to all elements of a given vector/matrix, returns the result'''
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    '''Applies the derivative of sigmoid to all elements of a given vector/matrix, returns the result'''
    return sigmoid(x) * (1-sigmoid(x))

# add random padding to an array of 2d-matrices
def add_padding(matrices, output_shape):
    ''' Add random padding to the matrices, to have the desired output shape, returns padded matrices '''
    dim, row, col = matrices.shape 
    out_row, out_col = output_shape 

    row_total_padding = out_row - row
    col_total_padding = out_col - col

    top_padding = randint(0, row_total_padding)
    left_padding = randint(0, col_total_padding)
    bottom_padding = row_total_padding - top_padding
    right_padding = col_total_padding - left_padding 

    return np.pad(matrices, ((0, 0), (top_padding, bottom_padding), (left_padding, right_padding)), mode='constant')

def get_prediction(probabilities):
    ''' Returns the index of the highest probability element from the probabilities vector '''
    if probabilities.max() < 0.1:
        return None
    
    return np.argmax(probabilities)


# accuracy function
def accuracy(predictions, true_result):
    '''
    Calculate the accuracy of predictions in comparaison with true result values
    Returns the accuracy calculated
    '''
    predictions = np.argmax(predictions, axis=1)
    true_result = np.argmax(true_result, axis=1)
    
    return np.mean(predictions == true_result)

#getter functions
def get_loss_function(name):
        """
        Returns the error function corresponding to the name given in parameter.
        Raise an exception if the given name does not correspond to any known function.

        Parameters
        ----------
            name : str
                name of the error function

        Return
        ------
            The error and its derivative functions
        """
        if name == 'mse':
            return mse, mse_prime
        else:
            raise Exception("Invalid error function")

def get_activation_function(name):
        """
        Returns the activation function corresponding to the name given in parameter.
        Raise an exception if the given name does not correspond to any known function.

        Parameters
        ----------
            name : str
                name of the activation function

        Return
        ------
            The activation and its derivative functions
        """
        if name == 'tanh':
            return tanh, tanh_prime
        elif name == 'sigmoid':
            return sigmoid, sigmoid_prime
        else:
            raise Exception("Invalid activation function")