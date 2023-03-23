import numpy as np

class MaxPooling:
    def __init__(self, kernel, stride, input_shape):
        self.kernel = kernel
        self.stride = stride

        (rowK, colK) = kernel.shape
        (row, col) = input_shape
        out_row, out_col = int(row/self.stride), int(col/self.stride)

        i0 = np.repeat(np.arange(rowK), colK)
        i1 = np.repeat(np.arange(0, row//2*2, self.stride), out_col)
        j0 = np.tile(np.arange(colK), rowK)
        j1 = np.tile(np.arange(0, col//2*2, self.stride), out_row)
        self.I = i0.reshape((-1, 1)) + i1.reshape((1, -1))
        self.J = j0.reshape((-1, 1)) + j1.reshape((1, -1))

    def forward_propagation(self, matrices):
        self.input_shape = matrices.shape

        (rowK, colK) = self.kernel.shape
        (dim, row, col) = matrices.shape
        out_row, out_col = int(row/self.stride), int(col/self.stride)

        trans_matrices = matrices[:, self.I, self.J]
        kernel_applied = self.kernel.reshape((-1, 1)) * trans_matrices
        output = kernel_applied.max(axis=1).reshape((dim, out_row, out_col))

        #saving information to be used in backpropagation
        self.maxI = np.argmax(kernel_applied, axis=1).reshape((dim, out_row, out_col))
        self.maxJ = np.tile(np.arange(kernel_applied.shape[2]), dim).reshape((dim, out_row, out_col))
        self.k = np.repeat(np.arange(dim), out_row*out_col).reshape((dim, out_row, out_col))

        return output

    def backpropagation(self, error, learning_rate):
        input_error = np.zeros(self.input_shape)

        rows, cols = self.I[self.maxI, self.maxJ], self.J[self.maxI, self.maxJ]
        input_error[self.k, rows, cols] = error

        return input_error

    def setActivationFunction(self, activation, activation_prime):
        return None

    def save_parameters(self, file):
        return None

    def load_parameters(self, file):
        return None
    
    def reset(self):
        return None