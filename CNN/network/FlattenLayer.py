import numpy as np

class FlattenLayer:
    def forward_propagation(self, matrices):
        matrices = matrices/128
        return matrices.reshape((1, -1))

    def backpropagation(self, error, learning_rate):
        return None

    def setActivationFunction(self, activation, activation_prime):
        return None
    
    def reset(self):
        return None