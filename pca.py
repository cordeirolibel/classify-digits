#========================================================
# Train to Classify Digits
# CordeiroLibel 2017
# PCA
#========================================================

from sklearn.decomposition import PCA
import numpy as np
import sys
from scipy.misc import toimage
import matplotlib.pyplot as plt

class Pca:

	def __init__(self, n_components):
		self.pca_skl = PCA(n_components=n_components)
		self.plots = 0 

	def fit(self,data):

		x = [d[0].T[0] for d in data]
		#y = [d[1].T[0] for d in data]

		self.pca_skl.fit(x)


	def transform(self,data, test = False):

		x = [d[0].T[0] for d in data]

		x = self.pca_skl.transform(x)

		# if is not a test data
		if not test:
			y = [d[1].T[0] for d in data]
			out = [(xi[:,None],yi[:,None]) for xi,yi in zip(x,y)]
		else:
			y = [d[1] for d in data]
			out = [(xi[:,None],yi) for xi,yi in zip(x,y)]

		return out

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

	def run(self,training_data,test_data):
		# fit
		print('pca fit...')
		sys.stdout.flush()
		self.fit(training_data)

		#transform
		print('pca transform...')
		sys.stdout.flush()
		training_data = self.transform(training_data)
		test_data = self.transform(test_data, test = True)

		print('pca end')
		sys.stdout.flush()
		return training_data, test_data

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
