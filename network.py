#========================================================
# Network to Classify Digits
# CordeiroLibel 2017
# Reference: http://neuralnetworksanddeeplearning.com/
#========================================================

# A module to implement the stochastic gradient descent learning
# algorithm for a feedforward neural network.  Gradients are calculated
# using backpropagation.  Note that I have focused on making the code
# simple, easily readable, and easily modifiable.  It is not optimized,
# and omits many desirable features.

#========================================================

import numpy as np
import random
import time

# ##### The Network Class

# The list ``sizes`` contains the number of neurons in the
# respective layers of the network.  For example, if the list
# was [2, 3, 1] then it would be a three-layer network, with the
# first layer containing 2 neurons, the second layer 3 neurons,
# and the third layer 1 neuron.  The biases and weights for the
# network are initialized randomly, using a Gaussian
# distribution with mean 0, and variance 1.  Note that the first
# layer is assumed to be an input layer, and by convention we
# won't set any biases for those neurons, since biases are only
# ever used in computing the outputs from later layers.

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.last_time  = time.time()

    #========================================================
    # ==>> Network output
    # Return the output of the network if "a" is input.
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    #========================================================
    # ==>> Stochastic Gradient Descent
    # Train the neural network using mini-batch stochastic
    # gradient descent.  The "training_data" is a list of tuples
    # "(x, y)" representing the training inputs and the desired
    # outputs.  The other non-optional parameters are
    # self-explanatory.  If "test_data" is provided then the
    # network will be evaluated against the test data after each
    # epoch, and partial progress printed out.  This is useful for
    # tracking progress, but slows things down substantially.
    def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):
        
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):

            # slice training_data in mini_batch_size size random
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
                
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))

    #========================================================
    # Update the network's weights and biases by applying
    # gradient descent using backpropagation to a single mini batch.
    # The "mini_batch" is a list of tuples "(x, y)", and "eta"
    # is the learning rate.

    def update_mini_batch(self, mini_batch, eta):
        
        #the same size of biases but of zeros
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]

    #========================================================
    # ==>> Backpropagation

    # Return a tuple (\nabla b, \nabla w) representing the
    # gradient for the cost function C_x.  \nabla b and
    # \nabla w are layer-by-layer lists of numpy arrays, similar
    # to self.biases and self.weights.

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y)*sigmoid(zs[-1],1)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the reference.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the reference, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid(z,1)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    # Return the number of test inputs for which the neural
    # network outputs the correct result. Note that the neural
    # network's output is assumed to be the index of whichever
    # neuron in the final layer has the highest activation.
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    # Return the vector of partial derivatives \partial C_x 
    # /\partial a for the output activations.
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    
    #function of time, milliseconds of last tic() 
    #reset define if you want clear the clock for the next tic()
    def tic(self,reset = True):
        toc = time.time()
        delta = toc - self.last_time
        if reset:
            self.last_time = toc
        return np.around(delta*1000, decimals = 2)

# Sigmoid function \sigma
def sigmoid(z, ordem = 0):
    if ordem is 0:
        return 1.0/(1.0+np.exp(-z))
    return sigmoid(z)*(1-sigmoid(z))



