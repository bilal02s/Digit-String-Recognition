import numpy as np
from random import randint

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

# add random padding to a 2d-matrix
def add_padding(matrix, output_shape):
    row_total_padding = output_shape[0] - matrix.shape[0]
    col_total_padding = output_shape[1] - matrix.shape[1]

    top_padding = randint(0, row_total_padding)
    left_padding = randint(0, col_total_padding)
    bottom_padding = row_total_padding - top_padding
    right_padding = col_total_padding - left_padding 

    return np.pad(matrix, ((top_padding, bottom_padding), (left_padding, right_padding)), mode='constant')