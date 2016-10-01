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

def verify(folder):
	for x in ['A','B','C','D','E','F','G','H','I','J']:

		try:
			with open(folder + '/' + x + '.pickle', 'r+b') as f:
				data = pickle.load(f)
		except Exception as e:
			print('Unable to read data to', folder + '/' + x + '.pickle', ':', e)
		number = data.shape[0]
		print(number)
# 	print(images[pic_number])
# 	print("\n\n\n\n")
# 	a = plt.imshow(images[pic_number])
# 	plt.show()

print("Printing for Train Data:-")
verify('notMNIST_large')
print("\nPrinting for Test Data:-")
verify('notMNIST_small')

# show_picture('notMNIST_large/A.pickle',0)
