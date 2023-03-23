import numpy as np

class MaxPooling:
    def __init__(self, kernel, stride, input_shape):
        self.kernel = kernel
        self.stride = stride

    def apply_kernel(self, matrix, kernel):
        (rowK, colK) = kernel.shape
        (row, col) = matrix.shape
        out_row, out_col = int(row/self.stride), int(col/self.stride)

        i0 = np.repeat(np.arange(rowK), colK)
        i1 = np.repeat(np.arange(0, row, self.stride), out_col)
        j0 = np.tile(np.arange(colK), rowK)
        j1 = np.tile(np.arange(0, col, self.stride), out_row)
        i = i0.reshape((-1, 1)) + i1.reshape((1, -1))
        j = j0.reshape((-1, 1)) + j1.reshape((1, -1))

        transformed_image = matrix[i, j]
        kernel_applied = kernel.reshape((-1, 1)) * transformed_image
        output = kernel_applied.max(axis=0).reshape((out_row, out_col))

        #saving information to be used in backpropagation
        maxI = np.argmax(kernel_applied, axis=0).reshape((out_row, out_col))
        maxJ = np.arange(kernel_applied.shape[1]).reshape((out_row, out_col))
        self.maxI.append(maxI)
        self.maxJ.append(maxJ)
        self.I.append(i)
        self.J.append(j)
        

        return output

    def forward_propagation(self, matrices):
        self.input_shape = matrices.shape
        self.maxI, self.maxJ, self.I, self.J = [], [], [], []
        output = []

        for matrix in matrices:
            output.append(self.apply_kernel(matrix, self.kernel))

        return np.array(output)

    def backpropagation(self, error, learning_rate):
        input_error = np.zeros(self.input_shape)

        for k in range(len(input_error)):
            rows, cols = self.I[k][self.maxI[k], self.maxJ[k]], self.J[k][self.maxI[k], self.maxJ[k]]
            input_error[k][rows, cols] = error[k]

        return input_error

    def setActivationFunction(self, activation, activation_prime):
        return None

    def save_parameters(self, file):
        return None

    def load_parameters(self, file):
        return None
    
    def reset(self):
        return None