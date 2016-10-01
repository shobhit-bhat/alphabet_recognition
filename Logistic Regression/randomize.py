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

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

def open_file(file_name):
	try:
		with open(file_name, 'r+b') as f:
			data = pickle.load(f)
	except Exception as e:
		print('Unable to read data to', file, ':', e)
	return data

def save_file(object, file_name):

	  print('Saving %s.' % file_name)
	  try:
		with open(file_name, 'wb') as f:
		  pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)
	  except Exception as e:
		print('Unable to save data to', file_name, ':', e)

train_dataset = np.zeros(shape = (200000, 28, 28))
train_labels = np.zeros(200000)
valid_dataset = np.zeros(shape = (100000, 28, 28))
valid_labels = np.zeros(100000)
test_dataset = np.zeros(shape = (100000, 28, 28))
test_labels = np.zeros(100000)

for x,obj in enumerate([valid_dataset, valid_labels, train_dataset, train_labels, test_dataset, test_labels]):
	obj = open_file('notMNIST_large/temp/' + str(x) + '.save')


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

for x,obj in enumerate([valid_dataset, valid_labels, train_dataset, train_labels, test_dataset, test_labels], start=10):
	save_file(obj,'notMNIST_large/temp/' + str(x) + '.save')