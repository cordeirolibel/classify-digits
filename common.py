#========================================================
# Common Functions
# CordeiroLibel 2017
# 
#========================================================
import time
import numpy as np 
import pandas as pd 

#========================================================
# Only x and y from DataFrame 
# Return 2 list
# 
def separate(df):
	x = df.filter(like='x').values.tolist()
	y = df.filter(like='y').values.tolist()

	x = np.array(x)
	y = np.array(y)

	return x,y

#========================================================
# Merge x and y lists  
# Return a dataframe
# 
def join(xs,ys):
	index = ['x'+str(i+1) for i in range(len(xs[0]))]
	index += ['y'+str(i+1) for i in range(len(ys[0]))]

	xy = [np.concatenate([x,y]) for x,y in zip(xs,ys)]

	df= pd.DataFrame(xy,columns=index)
	
	return df

#========================================================
#function of time, milliseconds of last tic() 
#reset define if you want clear the clock for the next tic()
last_time  = time.time()
def tic(reset = True):
    global last_time
    toc = time.time()
    delta = toc - last_time
    if reset:
        last_time = toc
    return np.around(delta*1000, decimals = 2)
