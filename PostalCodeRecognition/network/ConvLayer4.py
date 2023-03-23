import numpy as np

class ConvLayer:
    def __init__(self, shape, n_kernels, input_shape, padding='valid'):
        self.kernel_shape = shape
        self.n = n_kernels
        self.input_shape = input_shape
        self.kernels = np.array([np.random.randn(shape[0], shape[1]) for i in range(self.n)])
        self.bias = np.array([np.random.randn(1)[0] for i in range(self.n)])
        self.padding = padding

        #backpropagation params
        (rowK, colK) = self.kernel_shape
        (row, col) = self.input_shape

        if padding == 'same':
            i0 = np.repeat(np.arange(rowK), colK).reshape((-1, 1))
            i1 = np.repeat(np.arange(row), col).reshape((1, -1))
            j0 = np.tile(np.arange(colK), rowK).reshape((-1, 1))
            j1 = np.tile(np.arange(col), row).reshape((1, -1))
            self.I = i0 + i1
            self.J = j0 + j1
        elif padding == 'valid':
            i0 = np.repeat(np.arange(rowK), colK).reshape((-1, 1))
            i1 = np.repeat(np.arange(row-(rowK//2*2)), col-(colK//2*2)).reshape((1, -1))
            i2 = np.repeat(np.arange(row), col).reshape((1, -1))
            j0 = np.tile(np.arange(colK), rowK).reshape((-1, 1))
            j1 = np.tile(np.arange(col-(colK//2*2)), row-(rowK//2*2)).reshape((1, -1))
            j2 = np.tile(np.arange(col), row).reshape((1, -1))
            self.I, self.BKI = i0 + i1, i0 + i2
            self.J, self.BKJ = j0 + j1, j0 + j2
        else:
            raise Exception("Unvalid padding type")

    def forward_propagation(self, matrices):
        rowK, colK = self.kernel_shape
        dim, row, col = matrices.shape
        out_row, out_col = self.padding == 'same' and (row, col) or (row-(rowK//2*2), col-(colK//2*2))
        self.input_dim = dim

        if self.padding == 'same':
            matrices = np.pad(matrices, ((0, 0), (rowK//2, colK//2) , (rowK//2, colK//2)), mode='constant', constant_values=0)
        self.trans_matrices = matrices[:, self.I, self.J]
        self.conv_output = (np.matmul(self.kernels.reshape((self.n, -1)), self.trans_matrices)+self.bias.reshape(self.n, 1)).reshape((self.n*dim, out_row, out_col))
        output = self.activation(self.conv_output)

        return output

    def backpropagation(self, error, learning_rate):
        rowK, colK = self.kernel_shape 
        row, col = self.input_shape
        out_dim, out_row, out_col = error.shape

        common = error * self.activation_prime(self.conv_output)

        self.trans_matrices = self.trans_matrices.transpose(0, 2, 1)
        ker_err = np.matmul(
            common.reshape(out_dim, 1, out_row*out_col),
            np.concatenate([self.trans_matrices for i in range(self.n)], axis=1).reshape(out_dim, out_row*out_col, rowK*colK)
        ).reshape(self.input_dim, self.n, rowK*colK).sum(axis=0).reshape(self.n, rowK, colK)

        bias_err = common.sum(axis=(1, 2)).reshape(self.input_dim, self.n).sum(axis=0)

        if self.padding == 'same':
            err_padded = np.pad(common, ((0, 0), (rowK//2, colK//2) , (rowK//2, colK//2)), mode='constant', constant_values=0)
        elif self.padding == 'valid':
            err_padded = np.pad(common, ((0, 0), (rowK-1, colK-1) , (rowK-1, colK-1)), mode='constant', constant_values=0)
        input_err = np.matmul(
            np.concatenate([np.flip(self.kernels.reshape(self.n, 1, rowK*colK), axis=2) for i in range(self.input_dim)]),
            err_padded[:, self.BKI, self.BKJ]
        ).reshape(self.input_dim, self.n, row, col).sum(axis=1).reshape(self.input_dim, row, col)

        self.kernels -= learning_rate*ker_err 
        self.bias -= learning_rate*bias_err
        
        return input_err

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
