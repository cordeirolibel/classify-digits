#========================================================
# Train to Classify Digits
# CordeiroLibel 2017
# 
#========================================================
from common import tic
import mnist_loader as ml
from pca import Pca
import numpy as np 

tic()

#training_data:50000  validation_data:10000  test_data:10000 images
df_train, df_validation, df_test = ml.load_data_wrapper()#only_num = 3)
print(tic(),'ms')

df_train = ml.rotate_imgs(df_train,new=True)
print(tic(),'ms')

pca = Pca(50)
df_train,df_test,df_validation = pca.run([df_train,df_test,df_validation])
print(tic(),'ms')

names = ['train_rot_pca50','test_pca50','validation_pca50']
ml.save([df_train,df_test,df_validation],names = names)
print('end')


