import numpy as np
from Layer import Layer
from Network import Network
import util 

if __name__ == "__main__":
    #create the network
    net = Network()

    #setting the error and activation functions
    net.setErrorFunction(util.mse, util.mse_prime)
    net.setActivationFunction(util.tanh, util.tanh_prime)

    #adding two layers
    net.addLayer(Layer(2, 3))
    net.addLayer(Layer(3, 1))

    #creating the training data
    train = np.array([[[0, 0]],[[0, 1]],[[1, 0]],[[1, 1]]])
    finalResult = np.array([[[0]], [[1]], [[1]], [[0]]])

    #load parameters
    net.load_parameters("./params/XORparams")

    #making predictions
    predictions = net.predict(train)

    #display predictions
    for i in range(len(train)):
        print("data : " + str(train[i]) + ", expected : " + str(finalResult[i]) + ", predicted : " + str(predictions[i]))
