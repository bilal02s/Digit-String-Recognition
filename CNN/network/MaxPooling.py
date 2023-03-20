import numpy as np

def even(x):
    return (x//2)*2

class MaxPooling:
    def __init__(self, kernel, stride, input_shape):
        self.kernel = kernel
        self.stride = stride
        self.input_shape = input_shape

        (rowK, colK) = kernel.shape
        (row, col) = input_shape
        out_row, out_col = int(row/self.stride), int(col/self.stride)
        self.out_row, self.out_col = out_row, out_col

        row, col = even(row), even(col)

        i0 = np.repeat(np.arange(rowK), colK)
        i1 = np.repeat(np.arange(0, row, self.stride), out_col)
        j0 = np.tile(np.arange(colK), rowK)
        j1 = np.tile(np.arange(0, col, self.stride), out_row)
        self.transI = i0.reshape((-1, 1)) + i1.reshape((1, -1))
        self.transJ = j0.reshape((-1, 1)) + j1.reshape((1, -1))

    def forward_propagation(self, matrices):
        output = []

        trans_matrices = matrices[:, self.transI, self.transJ]
        output = (self.kernel.reshape((-1, 1)) * trans_matrices).max(axis=1).reshape((matrices.shape[0], self.out_row, self.out_col))

        return output

    def backpropagation(self, error, learning_rate):
        return None

    def setActivationFunction(self, activation, activation_prime):
        return None
    
    def reset(self):
        return None