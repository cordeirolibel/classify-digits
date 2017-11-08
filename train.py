#========================================================
# Train to Classify Digits
# CordeiroLibel 2017
# Reference: http://neuralnetworksanddeeplearning.com/
#========================================================
from network import Network, tic
import mnist_loader as ml
from pca import Pca
import numpy as np 

tic()

#training_data:50000  validation_data:10000  test_data:10000 images
df_train, df_validation, df_test = ml.load_data_wrapper()#only_num = 3)

#net = Network([784, 20, 10])
net = Network([20,40, 10])
net.cost_function = 'cross-entropy'
net.regularization_parameter = 5.0

print(tic(),'ms')

df_train = ml.more_data(df_train)
print(tic(),'ms')
exit()

pca = Pca(20)
training_data,test_data = pca.run(training_data,test_data)	
#pca.images_save()
print(tic(),'ms')

net.SGD(training_data, 10, 10, 0.1, test_data=test_data,plot_cost=True)
#net.weights_img_save()

#predict_data = net.predict(test_data)
#pca.plot(test_data,pause=False,name='Test Dataset')
#pca.plot(predict_data,name='Predict Dataset')
print(tic(),'ms')
print('end')


