import numpy as np

class Network:
    def __init__(self):
        self.layers = []

    def setErrorFunction(self, f, f_prime):
        self.f = f
        self.f_prime = f_prime

    def setActivationFunction(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        
    def addLayer(self, layer):
        layer.setActivationFunction(self.activation, self.activation_prime)
        self.layers.append(layer)

    def fit(self, train, result, generation=1000, learning_rate=0.1, printOn=100):
        n = len(train)
        error_value = 0

        for gen in range(generation):
            #error = np.zeros((1, len(result[0])))
            for i in range(n):
                output = train[i]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                error_value += self.f(output, result[i])
                error = self.f_prime(output, result[i])

                for layer in reversed(self.layers):
                    error = layer.backpropagation(error, learning_rate/(gen+1))

            error_value /= n

            if gen%printOn == 0:
                print("gen : " + str(gen) + ", error : " + str(error_value))

    def predict(self, toPredict):
        result = []

        for data in toPredict:
            output = data
            for layer in self.layers:
                output = layer.forward_propagation(output)

            result.append(output)

        return result

    def save_parameters(self, filename):
        file = open(filename, 'wb')
        
        for layer in self.layers:
            if hasattr(layer, "weights") and hasattr(layer, "weights"):
                layer.weights.tofile(file)
                layer.bias.tofile(file)

        file.close()

    def load_parameters(self, filename):
        file = open(filename, 'rb')

        for layer in self.layers:
            if (not hasattr(layer, "weights")) or (not hasattr(layer, "weights")):
                continue 

            weightsCount = layer.weights.size
            biasCount = layer.bias.size

            weights = np.fromfile(file, count=weightsCount).reshape(layer.weights.shape)
            bias = np.fromfile(file, count=biasCount).reshape(layer.bias.shape)

            layer.weights = weights
            layer.bias = bias

        