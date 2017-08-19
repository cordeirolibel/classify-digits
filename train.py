#========================================================
# Train to Classify Digits
# CordeiroLibel 2017
# Reference: http://neuralnetworksanddeeplearning.com/
#========================================================

import network
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper('data/mnist.pkl.gz')

net = network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 100.0, test_data=test_data)

print('end')



