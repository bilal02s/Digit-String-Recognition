import numpy as np

class ConvLayer:
    def __init__(self, shape, n_kernels, input_shape):
        self.kernel_shape = shape
        self.n = n_kernels
        self.input_shape = input_shape
        self.kernels = np.array([np.random.randn(shape[0], shape[1]) for i in range(self.n)])
        self.bias = np.array([np.random.randn(1)[0] for i in range(self.n)])

        #backpropagation params
        (rowK, colK) = self.kernel_shape
        (row, col) = self.input_shape

        i0 = np.repeat(np.arange(rowK), colK).reshape((-1, 1))
        i1 = np.repeat(np.arange(row), col).reshape((1, -1))
        j0 = np.tile(np.arange(colK), rowK).reshape((-1, 1))
        j1 = np.tile(np.arange(col), row).reshape((1, -1))
        self.I = i0 + i1
        self.J = j0 + j1
        '''self.transWI = []
        self.transWJ = []
        rowK, colK = self.kernel_shape 
        row, col = self.input_shape
        for x in range(self.kernel_shape[0]):
            for y in range(self.kernel_shape[1]):
                i0 = np.repeat(x, self.input_shape[1]).reshape(1, -1)
                i1 = np.arange(self.input_shape[0]).reshape(-1, 1)
                j0 = np.arange(self.input_shape[1]).reshape(1, -1) + y
                J = np.tile(j0, self.input_shape[0])
                I = i0 + i1
                self.transWI.append(I)
                self.transWJ.append(J)

        self.transWI = np.array(self.transWI).reshape(rowK, colK, row, col)
        self.transWJ = np.array(self.transWJ).reshape(rowK, colK, row, col)'''

    def transform_matrix(self, matrix):
        (rowK, colK) = self.kernel_shape
        (row, col) = matrix.shape
        matrix_padded = np.pad(matrix, (rowK//2, colK//2), mode='constant', constant_values=0)
        self.input.append(matrix_padded.copy())
        transformed_matrix = matrix_padded[self.I, self.J]

        return transformed_matrix

    def apply_kernel(self, matrix, kernel, bias, shape):
        row, col = shape
        bias = np.repeat(bias, row*col)
        output = np.dot(kernel.reshape((1, -1)), matrix) + bias
        output = output.reshape((row, col))

        return output

    def forward_propagation(self, matrices):
        self.input, self.input_transformed, self.conv_output = [], [], []
        output = []

        for matrix in matrices: 
            shape = matrix.shape 
            transformed_matrix = self.transform_matrix(matrix)
            self.input_transformed.append(transformed_matrix)

            for i in range(self.n):
                kernel, bias = self.kernels[i], self.bias[i]
                conv_output = self.apply_kernel(transformed_matrix, kernel, bias, shape)
                self.conv_output.append(conv_output)
                output.append(self.activation(conv_output))

        return np.array(output)

    def backpropagation(self, error, learning_rate):
        kernel_error = [[] for i in range(self.n)]
        bias_error = [[] for i in range(self.n)]
        input_error = [[] for i in range(len(self.input))]

        rowK, colK = self.kernel_shape 
        row, col = self.input_shape

        index = 0
        for m in range(len(self.input)):
            for k in range(self.n):
                common = error[index] * self.activation_prime(self.conv_output[index])

                ker_err = np.dot(common.reshape(1, -1), self.input_transformed[m].T).reshape(self.kernel_shape)
                err_padded = np.pad(common, (rowK//2, colK//2), mode='constant', constant_values=0)
                input_err = np.dot(np.flip(self.kernels[k].reshape(1, -1)), err_padded[self.I, self.J]).reshape(self.input_shape)
                #for i in range(rowK):
                 #   for j in range(colK):
                  #      ker_err[i][j] = self.input[m][self.transWI[i][j], self.transWJ[i][j]].sum()

                kernel_error[k].append(ker_err)
                bias_error[k].append(common.sum())
                input_error[m].append(input_err)

                index += 1

        total_ker_err = [sum(kernel_error[i]) for i in range(self.n)]
        total_bias_err = [sum(bias_error[i]) for i in range(self.n)]
        total_input_err = [sum(input_error[i]) for i in range(len(self.input))]

        for i in range(self.n):
            self.kernels[i] -= learning_rate*total_ker_err[i]
            self.bias[i] -=  learning_rate*total_bias_err[i]
        
        return np.array(total_input_err)

    def setActivationFunction(self, activation, activation_prime):
        self.activation = activation 
        self.activation_prime = activation_prime

    def save_parameters(self, file):
        self.kernels.tofile(file)
        self.bias.tofile(file)

    def load_parameters(self, file):
        kernelsCount = self.kernels.size
        biasCount = self.bias.size

        self.kernels = np.fromfile(file, count=kernelsCount).reshape(self.kernels.shape)
        self.bias = np.fromfile(file, count=biasCount).reshape(self.bias.shape)
    
    def reset(self):
        return None
