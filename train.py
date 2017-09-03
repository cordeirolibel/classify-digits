#========================================================
# Train to Classify Digits
# CordeiroLibel 2017
# Reference: http://neuralnetworksanddeeplearning.com/
#========================================================

import network
import mnist_loader as ml

#training_data:50000  validation_data:10000  test_data:10000 images
training_data, validation_data, test_data = ml.load_data_wrapper('data/mnist.pkl.gz')

#ml.digit_print(training_data[123])

net = network.Network([784, 30, 10])

net.tic()
net.SGD(training_data, 10, 10, 3.0, test_data=test_data)
net.weights_img_save()
print(net.tic(),'ms')

print('end')


