#========================================================
# Train to Classify Digits
# CordeiroLibel 2017
# Reference: http://neuralnetworksanddeeplearning.com/
#========================================================
from network import Network, tic
import mnist_loader as ml
from pca import Pca

#training_data:50000  validation_data:10000  test_data:10000 images
training_data, validation_data, test_data = ml.load_data_wrapper('data/mnist.pkl.gz')

#ml.digit_print(training_data[123])

#net = network.Network([784,30, 10])
net = Network([20,20, 10])

tic()

pca = Pca(20)
training_data,test_data = pca.run(training_data,test_data)

print(tic(),'ms')

net.SGD(training_data, 5, 10, 3.0, test_data=test_data)
#net.weights_img_save()

print(tic(),'ms')

print('end')


