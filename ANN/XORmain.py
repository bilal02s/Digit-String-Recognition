import numpy as np
from network.Layer import Layer
from network.Network import Network
import network.util as util

if __name__ == "__main__":
    #create the network
    net = Network()

    #adding two layers
    net.addLayer(Layer(2, 3, activation='sigmoid'))
    net.addLayer(Layer(3, 1, activation='sigmoid'))

    #creating the training data
    train = np.array([[[0, 0]],[[0, 1]],[[1, 0]],[[1, 1]]])
    finalResult = np.array([[[0]], [[1]], [[1]], [[0]]])

    #training the network
    net.fit(train, finalResult, loss='mse', generation=100, learning_rate=0.1)

    #making predictions
    predictions = net.predict(train)

    #display predictions
    for i in range(len(train)):
        print("data : " + str(train[i]) + ", expected : " + str(finalResult[i]) + ", predicted : " + str(predictions[i]))
        
    #save parameters
    net.save_parameters("./params/XORparams2")
