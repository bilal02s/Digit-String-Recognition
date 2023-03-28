import numpy as np
from .network import Layer, Network, util

if __name__ == "__main__":
    #create the network
    net = Network()

    #adding two layers
    net.setLayers([
        Layer(2, 3, activation='tanh'),
        Layer(3, 1, activation='tanh')
    ])

    #creating the training data
    train = np.array([[[-1, -1]],[[-1, 1]],[[1, -1]],[[1, 1]]])
    finalResult = np.array([[[-1]], [[1]], [[1]], [[-1]]])

    #training the network
    net.fit(train, finalResult, loss='mse', generation=1000, learning_rate=0.1)

    #making predictions
    predictions = net.predict(train)

    #display predictions
    for i in range(len(train)):
        print("data : " + str(train[i]) + ", expected : " + str(finalResult[i]) + ", predicted : " + str(predictions[i]))
        
    #save parameters
    net.save_parameters("./params/XORparams2")
