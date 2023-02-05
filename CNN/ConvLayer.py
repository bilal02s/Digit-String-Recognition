import numpy as np

class ConvLayer:
    def __init__(self, kernels):
        self.kernels = kernels
        self.n = len(kernels)

    def apply_kernel(self, matrix, kernel):
        (rowK, colK) = kernel.shape
        (row, col) = matrix.shape
        if row<4 or col<4:
            print((row, col))
        output = np.ndarray((row-rowK+1, col-colK+1))

        for j in range(row-rowK+1):
            for i in range(col-colK+1):
                output[j, i] = np.sum(np.multiply(matrix[j:j+rowK, i:i+colK], kernel))

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
