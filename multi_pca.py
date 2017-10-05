#========================================================
# CordeiroLibel 2017 - https://github.com/cordeirolibel/
# PCA for different number of components
#========================================================

from network import Network, tic
import mnist_loader as ml
from pca import Pca

#training_data:50000  validation_data:10000  test_data:10000 images
training_data, validation_data, test_data = ml.load_data_wrapper('data/mnist.pkl.gz')


#[1,2,...,19,20,1,2,...,49,50,1,2,...,99,100,102,104,...,298,300,305,310,...,775,780,784]
n_components_pca = list(range(1,21,1))
n_components_pca += list(range(1,51,1))
n_components_pca += list(range(1,100,1))
n_components_pca += list(range(100,300,2))
n_components_pca += list(range(300,784,5))
n_components_pca += [784]

# Create all PCAs
print('fit')
pcas = list()
for i in n_components_pca:
	print('  component',i)
	pcas.append(Pca(i))
	pcas[-1].fit(training_data)

# file to save the result
file = open('DadosPCAs','w')
file.write('n_components\trate(%)\ttime(ms)\n') 

# training with all PCAs
print('train')
for i,pca in zip(n_components_pca,pcas):
	print('==== Component',i,'====')
	# Preparing
	net = Network([i,20, 10])
	training_data_tmp = pca.transform(training_data)
	test_data_tmp = pca.transform(test_data, test = True)
	tic()

	# Train
	rate = net.SGD(training_data_tmp, 10, 10, 3.0, test_data=test_data_tmp)

	# Save
	line = str(i)+'\t'+str(rate)+'\t'+str(tic())
	print(line)
	file.write(line+'\n')

print('end')
file.close