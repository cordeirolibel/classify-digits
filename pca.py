#========================================================
# Train to Classify Digits
# CordeiroLibel 2017
# PCA
#========================================================

from sklearn.decomposition import PCA
from scipy.misc import toimage
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import sys

from common import separate, join

class Pca:

	def __init__(self, n_components):
		self.pca_skl = PCA(n_components=n_components)
		self.plots = 0 

	def fit(self,df):
		x,_ = separate(df)
		sys.stdout.flush()
		self.pca_skl.fit(x)


	def transform(self,df):

		x,y = separate(df)
		x = self.pca_skl.transform(x)
		return join(x,y)

	def images_save(self):
		size = int(np.sqrt(len(self.pca_skl.components_[0])))

		# save all components image
		k = 1
		for comp in self.pca_skl.components_:
			comp = comp.reshape(size,size)
			comp = (comp-np.min(comp))/(np.max(comp)-np.min(comp))
			img = toimage(comp)
			img.save('imgs/28x28/comp_'+str(k)+'.png')
			k+=1

		# save the mean image
		mean = self.pca_skl.mean_
		mean = mean.reshape(size,size)
		img = toimage(mean)
		img.save('imgs/28x28/mean'+'.png')

	def run(self,dfs):

		#force to list
		is_list = True
		if not type(dfs) is list:
			dfs = [dfs] 
			is_list = False

		df_all = pd.concat(dfs)

		# fit
		print('pca fit ...')
		sys.stdout.flush()
		self.fit(df_all)

		
		#transform
		print('pca transform ...')
		sys.stdout.flush()
		dfs_out = list()
		for df in dfs:
			dfs_out.append(self.transform(df))
		

		#return
		print('pca end')
		sys.stdout.flush()

		if is_list:
			return dfs_out
		else:
			return dfs_out[0]

	def plot(self, data,pause = True, name = 'Dataset'):
		n_points = 3000

		# ===> Data
		data = data[:n_points]#only n_points points
		colors = ['b','g','r','c','m','y','k','dimgrey','lime','darkred']

		x = [d[0].T[0] for d in data]

		if data[0][1].shape == (10,1):	#data is in one-hot
			one_hot = [d[1].T[0] for d in data]
			#one-hot to decimal
			y = [ np.where(num==1)[0][0] for num in one_hot]
		else:						#data is in decimal
			y = [d[1] for d in data]

		# ===> Plot
		self.plots+=1
		plt.figure(self.plots)

		# plot all points
		k = 1
		label = list(range(10))
		for pt,out in zip(x,y):
			plt.scatter(pt[0],pt[1],color=colors[out],alpha=.8,s=50,label=label[out])
			label[out] = '_nolegend_'
			if k%1000==0:
				print(k,'/',n_points,'Points')
			k+=1


		# show plot
		plt.title('PCA '+name)
		plt.legend(loc='best', shadow=False, scatterpoints=1)
		if pause:
			plt.show()
		else:
			plt.show(block = False)
