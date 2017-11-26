#========================================================
# Train to Classify Digits
# CordeiroLibel 2017
# 
#========================================================
from network import Network
from common import tic
import mnist_loader as ml
from pca import Pca
import numpy as np 

tic()

names = ['train_rot_pca50','test_pca50','validation_pca50']
df_train, df_validation, df_test = ml.load(names)
df_train = df_train[:2000]
print(tic(),'ms')

#net = Network([784, 40, 10])
net = Network([50,30, 10])
net.cost_function = 'cross-entropy'
#net.regularization_parameter = 5.0

net.SGD(df_train, 5, 10, 0.1, df_test=df_test,save_ws=True)#,plot_cost=True)

#net.weights_img_save()

#predict_data = net.predict(test_data)
print(tic(),'ms')
print('end')


