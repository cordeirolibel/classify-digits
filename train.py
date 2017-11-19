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
training_data, validation_data, test_data = ml.load_data_wrapper()#,only_num = 3)


#net = Network([784, 20, 10])
net = Network([100,100, 10])
net.cost_function = 'cross-entropy'
#net.regularization_parameter = 5.0

tic()

#test_data = ml.more_data(test_data)
#training_data = ml.more_data(training_data)

print(tic(),'ms')

pca = Pca(100)
training_data,test_data = pca.run(training_data,test_data)	

#pca.images_save()
print(tic(),'ms')
net.SGD(training_data, 100, 3, 0.03, test_data=test_data)#,plot_cost=True)
net.plot_w('after',color='orange')

#net.weights_img_save()


#predict_data = net.predict(test_data)
#pca.plot(test_data,pause=False,name='Test Dataset')
#pca.plot(predict_data,name='Predict Dataset')
print(tic(),'ms')
print('end')


