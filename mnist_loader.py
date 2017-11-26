# %load mnist_loader.py
"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
import poppy.zernike as zk
from scipy.misc import imrotate, toimage
import random
import pandas as pd 
from skimage.transform import pyramid_gaussian, pyramid_laplacian

from common import tic, separate


def load_data(dir):
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open(dir, 'rb')

    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")

    return (training_data, validation_data, test_data)

def load_data_wrapper(dir = 'data/mnist.pkl.gz', only_num = None, new=False):
    """Return a tuple containing 3 DataFrame"""
    print('Open files ...')
 
    try: #if exist csvs
        if new:
            erro
        df_train = pd.read_csv("data/train.csv")
        df_validation = pd.read_csv("data/validation.csv")
        df_test = pd.read_csv("data/test.csv")
        print("Load by","train.csv","validation.csv","test.csv")
    except:
        tr_d, va_d, te_d = load_data(dir)
        # ==> Array
        training_inputs   = [np.reshape(x, (784, 1)).T[0] for x in tr_d[0]]
        validation_inputs = [np.reshape(x, (784, 1)).T[0] for x in va_d[0]]
        test_inputs       = [np.reshape(x, (784, 1)).T[0] for x in te_d[0]]
        training_results   = [vectorized_result(y).T[0] for y in tr_d[1]]
        validation_results = [vectorized_result(y).T[0] for y in va_d[1]]
        test_results       = [vectorized_result(y).T[0] for y in te_d[1]]

        # ==> DataFrames
        # x
        index = ['x'+str(i+1) for i in range(784)]
        df_inputs_train      = pd.DataFrame(training_inputs,columns=index)
        df_inputs_validation = pd.DataFrame(validation_inputs,columns=index)
        df_inputs_test       = pd.DataFrame(test_inputs,columns=index)
        # y
        index = ['y'+str(i+1) for i in range(10)]
        df_results_train      = pd.DataFrame(training_results,columns=index)
        df_results_validation = pd.DataFrame(validation_results,columns=index)
        df_results_test      = pd.DataFrame(test_results,columns=index)
        # x+y
        df_train      = pd.concat([df_inputs_train,     df_results_train], axis=1)
        df_validation = pd.concat([df_inputs_validation,df_results_validation], axis=1)
        df_test       = pd.concat([df_inputs_test,      df_results_test], axis=1)

        #save
        df_train.to_csv("data/train.csv",index=False)
        df_validation.to_csv("data/validation.csv",index=False)
        df_test.to_csv("data/test.csv",index=False)


    # select
    if only_num != None:
        col = 'y'+str(only_num)
        df_train = df_train[df_train[col] == 1.0]

    return (df_train, df_validation, df_test)

#save 'dfs' with name 'names'
def save(dfs, names):

    if not type(dfs) is list:
        dfs = [dfs] 
        names = [names] 

    if len(dfs)!=len(names):
        print('Erro:', 'different size for \'dfs\' and \'names\'')
        exit()

    #save
    for df, name in zip(dfs, names):
        df.to_csv('data/'+name+'.csv',index=False)

#load a list of Dataframe with name 'names.csv'
def load(names):
    is_list = True
    if not type(names) is list:
        is_list = False
        names = [names] 

    #load
    dfs_out = list()
    for name in names:
        dfs_out.append(pd.read_csv('data/'+name+'.csv'))

    #return
    if is_list:
        return dfs_out
    else:
        return dfs_out[0]
        
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def digit_print(digit):
    #if know the number, print
    if len(digit) is 2:
        if digit[1].shape==():
            print(digit[1])
        else:
            print(digit[1].argmax())
        digit = digit[0]
    
    size = len(digit)
    l = int(np.sqrt(size))

    new_line = 0
    for pixel in digit:
        if len(pixel)==1: #vector image
            if pixel > 0.5:
                print('[]',end='')
            elif pixel > 0:
                print('::',end='')
            else:
                print('  ',end='')

            new_line+=1
            if new_line is l:
                print('|')
                new_line = 0
        else: #matrix image
            for p in pixel:
                if p > 0.5:
                    print('[]',end='')
                elif p > 0:
                    print('::',end='')
                else:
                    print('  ',end='')
            print('|')

# return more data
# name: to load in data/
# new=True: not load type_rot.cvs 
def rotate_imgs(df, ang = 10,name='train',new = False):

    print('Rotate Images ...')
    
    try: #if exist csvs
        if new:
            erro
        df = pd.read_csv("data/"+name+"_rot.csv")
        print('Load by',name+"_rot.csv")
    except:

        # slice
        xs,ys = separate(df)

        n_imgs = len(xs)
        imgs_new = list()
        k=0
        print(0,'/',n_imgs)
        for img,y in zip(xs,ys):
            #img = img.reshape([len(img),1])
            #y = y.reshape([len(y),1])
            y = y.tolist()

            k+=1
            if k%10000 == 0:
                print(k,'/',n_imgs)

            # array to matrix   
            size = int(np.sqrt(len(img)))
            imgM = np.reshape(img,(size,size))

            # rotate    
            img1M = imrotate(imgM,ang)/256
            img2M = imrotate(imgM,-ang)/256

            # matrix to array
            img1 = img1M.reshape(size**2,1).T[0].tolist()
            img2 = img2M.reshape(size**2,1).T[0].tolist()
            
            #save
            imgs_new.append(img1+y)
            imgs_new.append(img2+y)

        #save
        df_new = pd.DataFrame(imgs_new,columns = df.columns)
        imgs_new = [] #free memory
        df = pd.concat([df,df_new])
        df.to_csv("data/"+name+"_rot.csv",index=False)

    return df


def more_data(df, ang = 10,type='train',new = False):    
    # pyramid 
    #pyramid = tuple(pyramid_gaussian(imgM, downscale=2))
    pyramid = tuple(pyramid_laplacian(imgM, downscale=2))
    k=1
    for p in pyramid:
        toimage(p).save('imgs/pyramid/pl_'+str(k)+'.png')
        k+=1

    random.shuffle(data_out)
    return data_out


