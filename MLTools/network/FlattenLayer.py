import numpy as np

class FlattenLayer:
    def forward_propagation(self, matrices):
        ''' Flatten the input matrices, returns the flatten matrix '''
        self.input_shape = matrices.shape
        return matrices.reshape((1, -1))

    def backpropagation(self, error, learning_rate):
        ''' Reshape the error to the same shape as the input matrices, returns reshaped error '''
        return error.reshape(self.input_shape)

    def setActivationFunction(self, activation, activation_prime):
        return None

    def save_parameters(self, file):
        return None

    def load_parameters(self, file):
        return None
    
    def reset(self):
        return None