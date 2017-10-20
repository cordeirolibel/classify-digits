#========================================================
# Train to Classify Digits
# CordeiroLibel 2017
# Reference: http://neuralnetworksanddeeplearning.com/
#========================================================
from network import Network, tic
import mnist_loader as ml
from pca import Pca
import numpy as np 

#training_data:50000  validation_data:10000  test_data:10000 images
training_data, validation_data, test_data = ml.load_data_wrapper('data/mnist.pkl.gz')#,only_num = 3)

#net = Network([784, 20, 10])
net = Network([20,20, 10])
net.cost_function = 'cross-entropy'

tic()

pca = Pca(20)
training_data,test_data = pca.run(training_data,test_data)
#pca.images_save()
print(tic(),'ms')

net.SGD(training_data, 5, 10, 0.5, test_data=test_data)
#net.weights_img_save()


#predict_data = net.predict(test_data)
#pca.plot(test_data,pause=False,name='Test Dataset')
#pca.plot(predict_data,name='Predict Dataset')



print('end')


