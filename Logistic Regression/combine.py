from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

pickle_file = 'notMNIST.pickle'

train_dataset = np.zeros(shape = (200000, 28, 28))
train_labels = np.zeros(200000)
valid_dataset = np.zeros(shape = (100000, 28, 28))
valid_labels = np.zeros(100000)
test_dataset = np.zeros(shape = (100000, 28, 28))
test_labels = np.zeros(100000)

def open_file(file_name):
	try:
		with open(file_name, 'r+b') as f:
			data = pickle.load(f)
	except Exception as e:
		print('Unable to read data to', file, ':', e)
	return data

for x,obj in enumerate([valid_dataset, valid_labels, train_dataset, train_labels, test_dataset, test_labels]):
	obj = open_file('notMNIST_large/temp/' + str(x) + '.save')

try:
	f = open(pickle_file, 'wb')
  	save = {
	'train_dataset': train_dataset,
	'train_labels': train_labels,
	'valid_dataset': valid_dataset,
	'valid_labels': valid_labels,
	'test_dataset': test_dataset,
	'test_labels': test_labels,
	}
  	pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  	f.close()
except Exception as e:
  	print('Unable to save data to', pickle_file, ':', e)
  	raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)