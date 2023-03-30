import numpy as np

class MaxPooling:
    """
    A class representing a MaxPooling layer in a fully connected neural network,
    this layer accepts its kernels during initialisation, perform backpropagation to compute input error.

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
        Calculate indices to be used to compute error during backpropagation, saves them as attributes.
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
        self.input_shape = matrices.shape

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
        self.I = i0.reshape((-1, 1)) + i1.reshape((1, -1))
        self.J = j0.reshape((-1, 1)) + j1.reshape((1, -1))

        #transforming matrix and applying kernel
        trans_matrices = matrices[:, self.I, self.J]
        kernel_applied = self.kernel.reshape((-1, 1)) * trans_matrices
        output = kernel_applied.max(axis=1).reshape((dim, out_row, out_col))

        #saving information to be used in backpropagation
        self.maxI = np.argmax(kernel_applied, axis=1).reshape((dim, out_row, out_col))
        self.maxJ = np.tile(np.arange(kernel_applied.shape[2]), dim).reshape((dim, out_row, out_col))
        self.k = np.repeat(np.arange(dim), out_row*out_col).reshape((dim, out_row, out_col))

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