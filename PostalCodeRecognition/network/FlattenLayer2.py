import numpy as np

class FlattenLayer:
    def forward_propagation(self, matrices):
        self.input_shape = matrices.shape
        return matrices.reshape((1, -1))

    def backpropagation(self, error, learning_rate):
        return error.reshape(self.input_shape)

    def setActivationFunction(self, activation, activation_prime):
        return None

    def save_parameters(self, file):
        return None

    def load_parameters(self, file):
        return None
    
    def reset(self):
        return None