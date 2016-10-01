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

pixel_depth = 255.0

def show_picture(file, pic_number):
	try:
		with open(file, 'r+b') as f:
			data = pickle.load(f)
	except Exception as e:
		print('Unable to read data to', file, ':', e)
	images = (data * pixel_depth) + (pixel_depth / 2)
	print(images[pic_number])
	print("\n\n\n\n")
	a = plt.imshow(images[pic_number])
	plt.show()

# show_picture('notMNIST_large/A.pickle',1)
# show_picture('notMNIST_large/A.pickle',2)
# show_picture('notMNIST_large/A.pickle',3)

show_picture('notMNIST_large/temp/2.save',3)

# show_picture('notMNIST_large/temp/12.save',10203)
# show_picture('notMNIST_large/temp/12.save',133545)
# show_picture('notMNIST_large/temp/12.save',23224)
# show_picture('notMNIST_large/temp/12.save',33243)
# show_picture('notMNIST_large/temp/12.save',40000)
