#========================================================
# Train to Classify Digits
# CordeiroLibel 2017
# Reference: http://neuralnetworksanddeeplearning.com/
#========================================================
import network
import mnist_loader as ml
from pca import Pca

#training_data:50000  validation_data:10000  test_data:10000 images
training_data, validation_data, test_data = ml.load_data_wrapper('data/mnist.pkl.gz')

#ml.digit_print(training_data[123])
#exit()

net = network.Network([50,20, 10])

net.tic()

pca = Pca(50)
training_data,test_data = pca.run(training_data,test_data)

print(net.tic(),'ms')

net.SGD(training_data, 10, 10, 3.0, test_data=test_data)
#net.weights_img_save()

print(net.tic(),'ms')

print('end')


