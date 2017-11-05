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

def load_data_wrapper(dir = 'data/mnist.pkl.gz', only_num = None):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data(dir)
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    if only_num == None:
        training_data = list(zip(training_inputs, training_results))
    else:
        training_data = list()
        for inp, res in zip(training_inputs, training_results):
            if res[only_num] == [1]:
                training_data.append((inp,res))
    #training_data = list(training_data)
    #print(training_data[0:2])
    #training_data = list(zip(training_inputs, training_results))
    #print(training_data[0:2])
    #exit()

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

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

def more_data(data, ang = 10):

    print('more img')
    data_out = list()
    for img in data:
        num = list()
        if len(img) == 2:
            num = img[1]
            img = img[0]
        
        # array to matrix
        size = int(np.sqrt(len(img)))
        imgM = img.reshape(size,size)

        # rotate
        img1M = imrotate(imgM,ang)/256
        img2M = imrotate(imgM,-ang)/256

        # matrix to array
        img1 = img1M.reshape(size**2,1)
        img2 = img2M.reshape(size**2,1)

        # save
        if not num == []:
            img1 = (img1,num)
            img2 = (img2,num)
            img = (img,num)
        data_out.append(img1)
        data_out.append(img2)
        data_out.append(img)

    # pyramid 
    #pyramid = tuple(pyramid_gaussian(imgM, downscale=2))
    pyramid = tuple(pyramid_laplacian(imgM, downscale=2))
    k=1
    for p in pyramid:
        toimage(p).save('imgs/pyramid/pl_'+str(k)+'.png')
        k+=1

    random.shuffle(data_out)
    return data_out

#training_data, validation_data, test_data = load_data_wrapper('data/mnist.pkl.gz')

#img = training_data[222][0]

#size = int(np.sqrt(len(img)))
#img = img.reshape(size,size)


#print(zk.zernike(img))

#img = imrotate(img,45)/256

#print(zk.zernike(img))
