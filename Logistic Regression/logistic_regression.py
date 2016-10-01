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

def open_file(file_name):
	try:
		with open(file_name, 'r+b') as f:
			data = pickle.load(f)
	except Exception as e:
		print('Unable to read data to', file, ':', e)
	return data

data = open_file('notMNIST.pickle')

train_dataset = np.zeros()

for i in range(len(data.train_dataset)):
	train_dataset[i] = data.train_dataset[i,:,:]

# fit a logistic regression model to the data
model = LogisticRegression()
model.fit(data.train_dataset, data.train_label)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))