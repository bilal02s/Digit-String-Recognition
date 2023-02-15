import numpy as np

class MaxPooling:
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

    def apply_kernel(self, matrix, kernel):
        (rowK, colK) = kernel.shape
        (row, col) = matrix.shape
        out_row, out_col = int(row/self.stride), int(col/self.stride)

        #if (row%2 != 0):
         #   matrix = np.concatenate((matrix, np.array([[0 for i in range(col)]])))
        #if (col%2 != 0):
         #   matrix = np.concatenate((matrix, np.array([[0 for i in range(row+1)]]).reshape((12, 1))), axis=1)

        #output = np.ndarray(((row+1)//self.stride, (col+1)//self.stride))
        #(outputRow, outputCol) = output.shape

        #for j in range(outputRow):
         #   for i in range(outputCol):
          #      jm = j*self.stride
           #     im = i*self.stride
            #    output[j, i] = (matrix[jm:jm+rowK, im:im+colK]).max()

        i0 = np.repeat(np.arange(rowK), colK)
        i1 = np.repeat(np.arange(0, row, self.stride), out_col)
        j0 = np.tile(np.arange(colK), rowK)
        j1 = np.tile(np.arange(0, col, self.stride), out_row)
        i = i0.reshape((-1, 1)) + i1.reshape((1, -1))
        j = j0.reshape((-1, 1)) + j1.reshape((1, -1))

        transformed_image = matrix[i, j]
        output = (kernel.reshape((-1, 1)) * transformed_image).max(axis=0).reshape((out_row, out_col))

        return output

    def forward_propagation(self, matrices):
        output = []

        for matrix in matrices:
            output.append(self.apply_kernel(matrix, self.kernel))

        return output

    def backpropagation(self, error, learning_rate):
        return None

    def setActivationFunction(self, activation, activation_prime):
        return None