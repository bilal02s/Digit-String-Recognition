import numpy as np

class ConvLayer:
    def __init__(self, kernels):
        self.kernels = kernels
        self.n = len(kernels)

    def apply_kernel(self, matrix, kernel):
        (rowK, colK) = kernel.shape
        (row, col) = matrix.shape

        i0 = np.repeat(np.arange(rowK), colK).reshape((-1, 1))
        i1 = np.repeat(np.arange(row), col).reshape((1, -1))
        j0 = np.tile(np.arange(colK), rowK).reshape((-1, 1))
        j1 = np.tile(np.arange(col), row).reshape((1, -1))
        i = i0 + i1
        j = j0 + j1

        matrix_padded = np.pad(matrix, (rowK//2, colK//2), mode='constant', constant_values=0)
        transformed_image = matrix_padded[i, j]
        output = np.dot(kernel.reshape((1, -1)), transformed_image).reshape((row, col))

        return output

    def forward_propagation(self, matrices):
        output = []

        for matrix in matrices: 
            for kernel in self.kernels:
                output.append(self.apply_kernel(matrix, kernel))

        return output

    def backpropagation(self, error, learning_rate):
        return None

    def setActivationFunction(self, activation, activation_prime):
        return None
