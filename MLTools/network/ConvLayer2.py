import numpy as np
from . import util

class ConvLayer:
    """
    A class representing a convolution layer in a fully connected neural network,
    this convolution layer initialises its kernels randomly, and performs backpropagation to learn the values.

    ...

    Attributes
    ----------
    ker_shape : tuple
        tuple containing the shape of the kernels
    kernels : numpy.ndarray
        array containing the 2d kernels
    bias : numpy.ndarray
        array containing a bias value of every kernel
    padding : str, optional
        the type of padding to perform on input matrices
    activation : function
        the activation function
    activation_prime : function
        the derivative of the activation function

    Methods
    -------
    forward_propagation(input):
        performs a convolution operation on the input matrices/images using the defined kernels, returns the output result.

    backpropagation(error, learning_rate):
        performs backpropagation by updating the kernels using an error matrix and learning_rate value

    save_parameters(file):
        save this layer's kernels and bias into the open binary file given in parameter

    load_parameters(file):
        load parameters into this layer's kernels and bias from binary file given in parameter
    """
    def __init__(self, ker_shape, activation, padding='valid'):
        """
        Define the kernel's shape, initialise the kernels to random values, set the activation function
        and padding type.

        Parameters
        ----------
            ker_shape : triple
                the kernels dimension and shape
            activation : str
                the name of the activation function to use
            padding : str, optional
                the padding type, default type is valid
        """
        self.ker_shape = ker_shape
        self.padding = padding

        n, rowK, colK = ker_shape
        self.kernels = np.random.default_rng().normal(0, 1/np.sqrt(1764), size=ker_shape)
        self.bias = np.repeat(0.0, n).reshape(n)
        #self.kernels = [np.random.randn(n*rowK*colK).reshape(n, rowK, colK)
        #self.bias = np.random.randn(n).reshape(n)

        self.activation, self.activation_prime = util.get_activation_function(activation)

    def getTransIndices(self, in_shape):
        ''' 
        Computes transformation indices for the forward and back propagation.
        Returns the indices of the forward, and stores the indices of backward as attribut
        Parameters
        ----------
            in_shape : triplet
                The dimension and shape of input matrices whose transformation indices to be computed
        
        Returns
        -------
        I, J : matrices representing indices of rows and columns
        '''
        (n, rowK, colK) = self.ker_shape
        (dim, row, col) = in_shape

        if self.padding == 'same':
            i0 = np.repeat(np.arange(rowK), colK).reshape((-1, 1))
            i1 = np.repeat(np.arange(row), col).reshape((1, -1))
            j0 = np.tile(np.arange(colK), rowK).reshape((-1, 1))
            j1 = np.tile(np.arange(col), row).reshape((1, -1))
            I = i0 + i1
            J = j0 + j1
            self.BKI, self.BKJ = I, J
        elif self.padding == 'valid':
            i0 = np.repeat(np.arange(rowK), colK).reshape((-1, 1))
            i1 = np.repeat(np.arange(row-(rowK//2*2)), col-(colK//2*2)).reshape((1, -1))
            i2 = np.repeat(np.arange(row), col).reshape((1, -1))
            j0 = np.tile(np.arange(colK), rowK).reshape((-1, 1))
            j1 = np.tile(np.arange(col-(colK//2*2)), row-(rowK//2*2)).reshape((1, -1))
            j2 = np.tile(np.arange(col), row).reshape((1, -1))
            I, self.BKI = i0 + i1, i0 + i2
            J, self.BKJ = j0 + j1, j0 + j2
        else:
            raise Exception("Unvalid padding type")

        return I, J

    def forward_propagation(self, matrices):
        """
        Performs forward propagation, essentially a convolution operation for all the input matrices by each kernel individually.
        Returns the result of the convolution operation, the resulting third dimention is the product of the input dimension and the number of kernels.

        Parameters
        ----------
        input : numpy.ndarray
            a 3-dimensional matrix

        Returns
        -------
        output : numpy.ndarray
            a 3-dimensional matrix after being convolved by kernels
        """
        n, rowK, colK = self.ker_shape
        dim, row, col = matrices.shape
        out_row, out_col = self.padding == 'same' and (row, col) or (row-(rowK//2*2), col-(colK//2*2))
        
        self.input_shape = matrices.shape
        I, J = self.getTransIndices(matrices.shape)

        if self.padding == 'same':
            matrices = np.pad(matrices, ((0, 0), (rowK//2, rowK//2) , (colK//2, colK//2)), mode='constant', constant_values=0)
        self.trans_matrices = matrices[:, I, J]
        self.conv_output = (np.matmul(self.kernels.reshape((n, -1)), self.trans_matrices)+self.bias.reshape(n, 1)).reshape((n*dim, out_row, out_col))
        output = self.activation(self.conv_output)

        return output

    def backpropagation(self, error, learning_rate):
        """
        Performs backpropagation, and updating the kernels and bias accordingly
        calculates the error of the input used in the last forward propagation and returns it as an array
        of the same size as the input size

        Parameters
        ----------
        error : numpy.ndarray
            a 3-dimensional matrix error
        learning_rate : float
            the learning rate value

        Returns
        -------
        output : numpy.ndarray
            a 3-dimensional matrix representing the input error
        """
        n, rowK, colK = self.ker_shape 
        in_dim, row, col = self.input_shape
        out_dim, out_row, out_col = error.shape

        common = error * self.activation_prime(self.conv_output)

        self.trans_matrices = self.trans_matrices.transpose(0, 2, 1)
        ker_err = np.matmul(
            common.reshape(out_dim, 1, out_row*out_col),
            np.concatenate([self.trans_matrices for i in range(n)], axis=1).reshape(out_dim, out_row*out_col, rowK*colK)
        ).reshape(in_dim, n, rowK*colK).sum(axis=0).reshape(n, rowK, colK)

        bias_err = common.sum(axis=(1, 2)).reshape(in_dim, n).sum(axis=0)

        if self.padding == 'same':
            err_padded = np.pad(common, ((0, 0), (rowK//2, rowK//2) , (colK//2, colK//2)), mode='constant', constant_values=0)
        elif self.padding == 'valid':
            err_padded = np.pad(common, ((0, 0), (rowK-1, rowK-1) , (colK-1, colK-1)), mode='constant', constant_values=0)
        input_err = np.matmul(
            np.concatenate([np.flip(self.kernels.reshape(n, 1, rowK*colK), axis=2) for i in range(in_dim)]),
            err_padded[:, self.BKI, self.BKJ]
        ).reshape(in_dim, n, row, col).sum(axis=1).reshape(in_dim, row, col)

        self.kernels -= learning_rate*ker_err 
        self.bias -= learning_rate*bias_err
        
        return input_err

    def save_parameters(self, file):
        """
        Saves this layer's parameters (kernels and bias) into the open file given in parameter

        Parameters
        ----------
        file : open file
            an open file to write
        """
        self.kernels.tofile(file)
        self.bias.tofile(file)

    def load_parameters(self, file):
        """
        Loads from the open file given in parameter, the matrices that make up this layer's parameters (kernels and bias)

        Parameters
        ----------
        file : open file
            an open file to read
        """
        kernelsCount = self.kernels.size
        biasCount = self.bias.size

        self.kernels = np.fromfile(file, count=kernelsCount).reshape(self.kernels.shape)
        self.bias = np.fromfile(file, count=biasCount).reshape(self.bias.shape)
    
    def reset(self):
        return None
