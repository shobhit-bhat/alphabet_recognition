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

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def make_arrays(nb_rows, img_size):
  if nb_rows:
	dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
	labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
	dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
	
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
	try:
	  with open(pickle_file, 'rb') as f:
		letter_set = pickle.load(f)
		# let's shuffle the letters to have random validation and training set
		np.random.shuffle(letter_set)
		if valid_dataset is not None:
		  valid_letter = letter_set[:vsize_per_class, :, :]
		  valid_dataset[start_v:end_v, :, :] = valid_letter
		  valid_labels[start_v:end_v] = label
		  start_v += vsize_per_class
		  end_v += vsize_per_class
					
		train_letter = letter_set[vsize_per_class:end_l, :, :]
		train_dataset[start_t:end_t, :, :] = train_letter
		train_labels[start_t:end_t] = label
		start_t += tsize_per_class
		end_t += tsize_per_class
	except Exception as e:
	  print('Unable to process data from', pickle_file, ':', e)
	  raise
	
  return valid_dataset, valid_labels, train_dataset, train_labels
			
def save_file(object, file_name):

	  print('Saving %s.' % file_name)
	  try:
		with open(file_name, 'wb') as f:
		  pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)
	  except Exception as e:
		print('Unable to save data to', file_name, ':', e)
			

train_size = 200000
valid_size = 10000
test_size = 10000

train_datasets = ['notMNIST_large/A.pickle','notMNIST_large/B.pickle','notMNIST_large/C.pickle','notMNIST_large/D.pickle','notMNIST_large/E.pickle',
				  'notMNIST_large/F.pickle','notMNIST_large/G.pickle','notMNIST_large/H.pickle','notMNIST_large/I.pickle','notMNIST_large/J.pickle']

test_datasets = ['notMNIST_small/A.pickle','notMNIST_small/B.pickle','notMNIST_small/C.pickle','notMNIST_small/D.pickle','notMNIST_small/E.pickle',
				  'notMNIST_small/F.pickle','notMNIST_small/G.pickle','notMNIST_small/H.pickle','notMNIST_small/I.pickle','notMNIST_small/J.pickle']

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

for x,obj in enumerate([valid_dataset, valid_labels, train_dataset, train_labels, test_dataset, test_labels]):
	save_file(obj,'notMNIST_large/temp/' + str(x) + '.save')

# print("Train Labels\n", train_labels)
# print("\n\nTest Labels\n", test_labels)
# print("\n\nValid Labels\n", valid_labels)


print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)