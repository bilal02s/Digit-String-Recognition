import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.input_size = input_size
        self.output_size = output_size

    def reset(self):
        self.weights = np.random.rand(self.input_size, self.output_size) - 0.5
        self.bias = np.random.rand(1, self.output_size) - 0.5

    def getInputSize(self):
        return self.input_size

    def getOutputSize(self):
        return self.output_size

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
        common = error * self.activation_prime(self.output)
        self.bias_err = common
        self.input_err = np.dot(common, self.weights.T)
        self.weights_err = np.dot(self.input.T, common)
        
        #adjusting the parameters
        self.weights -= learning_rate * self.weights_err
        self.bias -= learning_rate * self.bias_err

        return self.input_err

    def save_parameters(self, file):
        self.weights.tofile(file)
        self.bias.tofile(file)

    def load_parameters(self, file):
        weightsCount = self.weights.size
        biasCount = self.bias.size

        self.weights = np.fromfile(file, count=weightsCount).reshape(self.weights.shape)
        self.bias = np.fromfile(file, count=biasCount).reshape(self.bias.shape)
        