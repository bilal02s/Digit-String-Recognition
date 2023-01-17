import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) 
        self.bias = np.random.rand(1, output_size) 

    def setActivationFunction(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        self.activationOutput = self.activation(self.output)

        return self.activationOutput

    def backpropagation(self, error, learning_rate):
        # finding (dE/dz) * f'(y)
        #print(error)
        #print(self.activation_prime(self.output))
        common = error * self.activation_prime(self.output)
        self.bias_err = common
        self.input_err = np.dot(common, self.weights.T)
        self.weights_err = np.dot(self.input.T, common)
        
        #adjusting the parameters
        self.weights -= learning_rate * self.weights_err
        self.bias -= learning_rate * self.bias_err

        return self.input_err
        