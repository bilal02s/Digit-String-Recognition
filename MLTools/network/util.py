import numpy as np

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