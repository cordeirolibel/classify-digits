#========================================================
# Train to Classify Digits
# CordeiroLibel 2017
# PCA
#========================================================

from sklearn.decomposition import PCA
import numpy as np
import sys

class Pca:

	def __init__(self, n_components):
		self.pca_skl = PCA(n_components=n_components)


	def fit(self,data):

		x = [d[0].T[0] for d in data]
		#y = [d[1].T[0] for d in data]

		self.pca_skl.fit(x)


	def transform(self,data, test = False):

		x = [d[0].T[0] for d in data]

		x = self.pca_skl.transform(x)

		if not test:
			y = [d[1].T[0] for d in data]
			out = [(xi[:,None],yi[:,None]) for xi,yi in zip(x,y)]
		else:
			y = [d[1] for d in data]
			out = [(xi[:,None],yi) for xi,yi in zip(x,y)]

		return out

	def run(self,training_data,test_data):

		print('pca fit...')
		sys.stdout.flush()

		self.fit(training_data)

		print('pca transform...')
		sys.stdout.flush()

		training_data = self.transform(training_data)
		test_data = self.transform(test_data, test = True)

		print('pca end')
		sys.stdout.flush()

		return training_data, test_data