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
net = Network([40,50, 10])
net.cost_function = 'cross-entropy'
net.regularization_parameter = 1.0
tic()

pca = Pca(40)
training_data,test_data = pca.run(training_data,test_data)
#pca.images_save()
print(tic(),'ms')

net.SGD(training_data, 20, 10, 0.5, test_data=test_data)#,plot_cost=True)
#net.weights_img_save()


#predict_data = net.predict(test_data)
#pca.plot(test_data,pause=False,name='Test Dataset')
#pca.plot(predict_data,name='Predict Dataset')

print('end')


