import numpy as np

class ConvLayer:
    """
    A class representing a convolution layer in a fully connected neural network,
    this convolution layer accepts its kernels during initialisation, does not perform backpropagation.

    ...

    Attributes
    ----------
    kernels : numpy.ndarray
        array containing the 2d kernels

    Methods
    -------
    forward_propagation(input):
        performs a convolution operation on the input matrices/images using the defined kernels, returns the output result.
    """
    def __init__(self, kernels):
        """
        Set the kernels to the attribut kernels of the layer

        Parameters
        ----------
            kernels : numpy.ndarray
                array containing the 2d kernels
        
        """
        self.kernels = np.array(kernels)

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
        output = []

        (n, rowK, colK) = self.kernels.shape
        (dim, row, col) = matrices.shape

        #calculate transformation indices
        i0 = np.repeat(np.arange(rowK), colK).reshape((-1, 1))
        i1 = np.repeat(np.arange(row), col).reshape((1, -1))
        j0 = np.tile(np.arange(colK), rowK).reshape((-1, 1))
        j1 = np.tile(np.arange(col), row).reshape((1, -1))
        I = i0 + i1
        J = j0 + j1
        
        #pad and transform matrices, and then apply kernels.
        matrices_padded = np.pad(matrices, ((0, 0), (rowK//2, rowK//2) , (colK//2, colK//2)), mode='constant', constant_values=0)
        trans_matrices = matrices_padded[:, I, J]
        output = np.matmul(self.kernels.reshape((n, -1)), trans_matrices).reshape((n*dim, row, col))

        return output

    def backpropagation(self, error, learning_rate):
        return None

    def save_parameters(self, file):
        return None

    def load_parameters(self, file):
        return None
    
    def reset(self):
        return None
