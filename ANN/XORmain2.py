import numpy as np
from network.Layer import Layer
from network.Network import Network
import network.util as util

if __name__ == "__main__":
    #create the network
    net = Network()

    #adding two layers
    net.setLayers([
        Layer(2, 3, activation='tanh'),
        Layer(3, 1, activation='tanh')
    ])

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
