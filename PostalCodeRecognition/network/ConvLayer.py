import numpy as np

class ConvLayer:
    def __init__(self, kernels, input_shape):
        self.kernels = np.array(kernels)
        self.n = len(kernels)
        self.input_shape = input_shape
        self.kernel_shape = kernels[0].shape

        (rowK, colK) = self.kernel_shape
        (row, col) = self.input_shape

        i0 = np.repeat(np.arange(rowK), colK).reshape((-1, 1))
        i1 = np.repeat(np.arange(row), col).reshape((1, -1))
        j0 = np.tile(np.arange(colK), rowK).reshape((-1, 1))
        j1 = np.tile(np.arange(col), row).reshape((1, -1))
        self.transI = i0 + i1
        self.transJ = j0 + j1

    def forward_propagation(self, matrices):
        output = []

        rowK, colK = self.kernel_shape
        dim, row, col = matrices.shape

        matrices_padded = np.pad(matrices, ((0, 0), (rowK//2, colK//2) , (rowK//2, colK//2)), mode='constant', constant_values=0)
        trans_matrices = matrices_padded[:, self.transI, self.transJ]
        output = np.dot(self.kernels.reshape((self.n, -1)), trans_matrices).reshape((self.n*dim, row, col))

        return output

    def backpropagation(self, error, learning_rate):
        return None

    def setActivationFunction(self, activation, activation_prime):
        return None
    
    def reset(self):
        return None
