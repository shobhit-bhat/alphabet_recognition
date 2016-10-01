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

# Config the matlotlib backend as plotting inline in IPython

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
						 dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
	image_file = os.path.join(folder, image)
	try:
	  image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
	  if image_data.shape != (image_size, image_size):
		raise Exception('Unexpected image shape: %s' % str(image_data.shape))
	  dataset[num_images, :, :] = image_data
	  num_images = num_images + 1
	except IOError as e:
	  print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
	
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
	raise Exception('Many fewer images than expected: %d < %d' %
					(num_images, min_num_images))
	
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset[0]))
  print('Standard deviation:', np.std(dataset[0]))
  return dataset
		
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
	set_filename = folder + '.pickle'
	dataset_names.append(set_filename)
	if os.path.exists(set_filename) and not force:
	  # You may override by setting force=True.
	  print('%s already present - Skipping pickling.' % set_filename)
	else:
	  print('Pickling %s.' % set_filename)
	  dataset = load_letter(folder, min_num_images_per_class)
	  try:
		with open(set_filename, 'wb') as f:
		  pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
	  except Exception as e:
		print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(['notMNIST_large/A', 'notMNIST_large/B', 'notMNIST_large/C', 'notMNIST_large/D', 'notMNIST_large/E', 'notMNIST_large/F', 'notMNIST_large/G', 'notMNIST_large/H', 'notMNIST_large/I', 'notMNIST_large/J'], 45000)
test_datasets = maybe_pickle(['notMNIST_small/A', 'notMNIST_small/B', 'notMNIST_small/C', 'notMNIST_small/D', 'notMNIST_small/E', 'notMNIST_small/F', 'notMNIST_small/G', 'notMNIST_small/H', 'notMNIST_small/I', 'notMNIST_small/J'], 1800)