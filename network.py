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
from bokeh.plotting import figure, show, output_file
import random
import sys
from scipy.misc import toimage
import pandas as pd 

from common import separate, join

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

    def __init__(self, sizes, large_weight = False):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        if large_weight:
            self.weights = [np.random.randn(y, x) 
                            for x, y in zip(sizes[:-1], sizes[1:])]
        else:
            self.weights = [np.random.randn(y, x)/np.sqrt(x) 
                            for x, y in zip(sizes[:-1], sizes[1:])]
        self.cost_function = 'quadratic' 
        #self.cost_function = 'cross-entropy'
        self.custom_cost_function = np.ones((sizes[-1],1))
        self.regularization_parameter = 0
        self.n_train = None

        #dataframe for weights
        self.n_weights = sum([sizes[i]*sizes[i+1] for i in range(self.num_layers-1)])
        index = ['w'+str(i+1) for i in range(self.n_weights)]
        self.df_weights = pd.DataFrame(columns=index)

    #========================================================
    # ==>> Network output
    # Return the output of the network if "x" is input.
    def feedforward(self, a):
        a = a.reshape([len(a),1])
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a)+b
            a = sigmoid(z)
            
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
    def SGD(self, df_train, epochs, mini_batch_size, eta,df_test=None, plot_cost = False,save_ws = False):
        
        sys.stdout.flush()

        if df_test is not None: 
            n_test = len(df_test)
        self.n_train = len(df_train)

        costs_test = []
        costs_train = []

        for j in range(epochs):

            if df_test is not None: 
                print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(df_test), n_test))
                if plot_cost:
                    costs_test.append(self.cost_func(df_test,convert=True))
                    costs_train.append(self.cost_func(df_train))
            else:
                print ("Epoch {0} complete".format(j))
            sys.stdout.flush()

            # slice training_data in mini_batch_size size random
            df_train = df_train.sample(frac=1).reset_index(drop=True)
     
            mini_batches = [
                df_train[k:k+mini_batch_size]
                for k in range(0, self.n_train, mini_batch_size)]

            weights = list()
            for df_mini_batch in mini_batches:
                self.update_mini_batch(df_mini_batch, eta)

            if save_ws:
                # weights to array
                w_array = np.array([])
                for ws in self.weights:
                    w_array = np.append(ws.ravel(),w_array)
                weights.append(w_array.tolist())
            if save_ws:
                index = ['w'+str(i+1) for i in range(self.n_weights)]
                weights = pd.DataFrame(weights,columns=index)
                self.df_weights = pd.concat([self.df_weights, weights])
                
    
        rate = 0
        if df_test is not None:
            rate = self.evaluate(df_test)
            print ("Epoch {0}: {1} / {2}".format(epochs, rate, n_test))            
        else:
            print ("Epoch {0} complete".format(epochs))
            n_test=1

        sys.stdout.flush()

        if plot_cost:
            self.plotCost(costs_test,'Test')
            self.plotCost(costs_train,'Train')
        return rate*100/n_test
        
    #========================================================
    # Update the network's weights and biases by applying
    # gradient descent using backpropagation to a single mini batch.
    # The "mini_batch" is a list of tuples "(x, y)", and "eta"
    # is the learning rate.

    def update_mini_batch(self, df_mini_batch, eta):

        xs, ys = separate(df_mini_batch)

        #the same size of biases but of zeros
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in zip(xs,ys):
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        rp = self.regularization_parameter
        n = self.n_train
        self.weights = [(1-eta*rp/n)*w-(eta/len(df_mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(df_mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]

    #========================================================
    # ==>> Backpropagation
    # Return a tuple (\nabla b, \nabla w) representing the
    # gradient for the cost function C_x.  \nabla b and
    # \nabla w are layer-by-layer lists of numpy arrays, similar
    # to self.biases and self.weights.

    def backprop(self, x, y):

        # x, a[0]=layer1, a[1]=layer2, ...
        a = list()
        z = list()

        #column vector
        x = x.reshape([len(x),1])
        y = y.reshape([len(y),1])

        for layer in range(self.num_layers-1):
            # z = w*a+b
            if layer == 0:
                z.append(np.dot(self.weights[layer],     x    ) + self.biases[layer])
            else:
                z.append(np.dot(self.weights[layer],a[layer-1]) + self.biases[layer])
            # a = sig(z)
            a.append(sigmoid(z[-1]))

        # BP1: d = partial C/partial a^L * sigmoid'(z^L)
        #      d = (a^L-y) (*) sigmoid'(z^L)
        # (*) Hadamart Product -> *
        
        if self.cost_function == 'quadratic':
            delta = (a[-1] - y) * (sigmoid(z[-1],1))
        else: #cross-entropy
            delta = (sigmoid(z[-1],1)) * ((1-y)/(1-a[-1]) - y/a[-1])
        
        # same size of biases but of zeros
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
 

        nabla_b[-1] = delta
        #nabla_w[-1] = np.dot(a[-2],delta)
        nabla_w[-1] = np.dot(delta, a[-2].T)

        #if num_layers=4   
        #layer <- 1 <- 0 
        for layer in range(self.num_layers-3,-1,-1):
            
            # d = (wT*d) (*) sigmoid'(z)
            # d^l = ((w^(l+1))^T*d^(l+1)) (*) sigmoid(z^l)
            # (*) Hadamart Product -> *
            aux = np.dot(self.weights[layer+1].T,delta)
            delta = aux * sigmoid(z[layer],1)

            nabla_b[layer] = delta
            #nabla_w[layer] = np.dot(a[layer-1],delta)
            if layer == 0:
                nabla_w[layer] = np.dot(delta,     x.T     )
            else:
                nabla_w[layer] = np.dot(delta, a[layer-1].T)

        return (nabla_b, nabla_w)


    #========================================================
    # Return the number of test inputs for which the neural
    # network outputs the correct result. Note that the neural
    # network's output is assumed to be the index of whichever
    # neuron in the final layer has the highest activation.
    def evaluate(self, df_test):
        xs, ys = separate(df_test)
        test_results = list()

        for x, y in zip(xs, ys):
            a = self.feedforward(x)
            test_results.append((np.argmax(a), np.argmax(y)))

        return sum(int(x == y) for (x, y) in test_results)

    #predict data
    def predict(self, test_data):
        predict_data = [(x,np.argmax(self.feedforward(x)))
                        for (x, y) in test_data]
        return predict_data

    

    #========================================================
    #'quadratic' 
    #'cross-entropy'
    def cost_func(self,data, convert = False):

        test_results = [(self.feedforward(x), y)
                        for (x, y) in data]

        n = len(data)
        rp = self.regularization_parameter
        if self.cost_function == 'quadratic':
            cost = 0
            for (a, y) in test_results:
                if convert:
                    y = vectorized(y, len(a))
                e = np.linalg.norm(a-y)
                cost+=e**2/(2*n)
        else:#cross_entropy
            cost = 0
            for (a, y) in test_results:
                if convert:
                    y = vectorized(y, len(a))
                e = y*np.log(a)+(1-y)*np.log(1-a)
                cost+=-np.sum(np.nan_to_num(e))/n

        for w in self.weights:

            w = np.linalg.norm(w)
            cost+=rp*w**2/(2*n)

        return float(cost)

    def plotCost(self,costs, name = 'cost'):


        x = list(range(len(costs)))

        output_file('imgs/'+name+'.html')
        p = figure(title="Cost Function: "+name,x_range=(0, x[-1]*1.05), y_range=(min(costs)*0.8, max(costs[1:])*1.2) )
        p.line(x,costs, line_width=3)

        show(p) 
    #========================================================
    # Return the vector of partial derivatives \partial C_x 
    # /\partial a for the output activations.
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)


    #========================================================

    def weights_img_save(self):

        n_imgs = self.weights[1].shape[0]
        n_inputs = self.weights[0].shape[1]

        imgs = np.dot(self.weights[1],self.weights[0])

        k = 0
        size = int(np.sqrt(self.weights[0][0].shape))
        for img in imgs:
            #array to matrix
            img = img.reshape(size,size)

            #normalize
            img -= img.min()
            img /= img.max()

            #save imgs
            img = toimage(img)
            img.save('imgs/28x28/w_'+str(k)+'.png')
            k += 1


def vectorized(num, size):
    try:
        if size == len(num):
            return num
    except: None
    e = np.zeros((size, 1))
    e[num] = 1.0
    return e

#========================================================
# Sigmoid function \sigma
# z is a numpy array
def sigmoid(z, ordem = 0):
    if ordem is 0:
        return 1.0/(1.0+np.exp(-z))
    return sigmoid(z)*(1-sigmoid(z))

#========================================================
# softmax function 
# z is a numpy array
def softmax(z):
    total = sum(z)
    return z/total

