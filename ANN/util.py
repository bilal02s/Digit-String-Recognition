import numpy as np

# Mean Squared Error and its derivative
def mse(prediction, expected):
    return np.mean(np.power(prediction - expected, 2))

def mse_prime(prediction, expected):
    return 2*(prediction - expected)/expected.size

# activation function and its derivative
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;