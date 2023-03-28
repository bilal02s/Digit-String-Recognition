import numpy as np

class MaxPooling:
    """
    A class representing a MaxPooling layer in a fully connected neural network,
    this layer accepts its kernel during initialisation, does not perform backpropagation.

    ...

    Attributes
    ----------
    kernel : numpy.ndarray
        The 2d kernel

    Methods
    -------
    forward_propagation(input):
        performs Max pooling operation on the input matrices/images using the defined kernel, returns the output result.
    """
    def __init__(self, kernel, stride):
        """
        Set the kernel to the attribut kernel of the layer, as well as the stride value

        Parameters
        ----------
            kernels : numpy.ndarray
                The 2d kernels
            stride : int
                Integer value representing the stride
        """
        self.kernel = kernel
        self.stride = stride

    def forward_propagation(self, matrices):
        """
        Performs forward propagation, essentially a maxpooling operation for all the input matrices by the kernel.
        Returns the result of the maxpooling operation, the resulting third dimention is the same as the input dimension.

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

        #finding input/output dimensions
        (rowK, colK) = self.kernel.shape
        (dim, row, col) = matrices.shape
        out_row, out_col = int(row/self.stride), int(col/self.stride)
        trim_row, trim_col = row//self.stride*self.stride, col//self.stride*self.stride

        #calculating transformation indices
        i0 = np.repeat(np.arange(rowK), colK)
        i1 = np.repeat(np.arange(0, trim_row, self.stride), out_col)
        j0 = np.tile(np.arange(colK), rowK)
        j1 = np.tile(np.arange(0, trim_col, self.stride), out_row)
        I = i0.reshape((-1, 1)) + i1.reshape((1, -1))
        J = j0.reshape((-1, 1)) + j1.reshape((1, -1))

        #transforming matrix and applying kernel
        trans_matrices = matrices[:, I, J]
        output = (self.kernel.reshape((-1, 1)) * trans_matrices).max(axis=1).reshape((matrices.shape[0], out_row, out_col))

        return output

    def backpropagation(self, error, learning_rate):
        return None

    def setActivationFunction(self, activation, activation_prime):
        return None

    def save_parameters(self, file):
        return None

    def load_parameters(self, file):
        return None
    
    def reset(self):
        return None