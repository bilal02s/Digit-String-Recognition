import numpy as np

class FlattenLayer:
    def forward_propagation(self, matrices):
        matrices = [(matrix/128)-1.0 for matrix in matrices]
        output = np.concatenate(matrices)
        return output.reshape((1, -1))

    def backpropagation(self, error, learning_rate):
        return None

    def setActivationFunction(self, activation, activation_prime):
        return None
    
    def reset(self):
        return None