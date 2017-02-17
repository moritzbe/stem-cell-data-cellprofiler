from data_tools import *
from algorithms import *
from plot_lib import *
# from nets import *
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn import preprocessing
import numpy as np
import code 
import math
from random import shuffle


# Loading the data into X, y
DATA = loadnumpy("data_no_zeros.npy").astype(np.float64)
ground_truth = loadnumpy("labels_no_zeros.npy").astype(int)


class_names = ["0", "1", "2", "3"]

print "The test data will be the data of exp 2."
X_train = DATA
y_train = ground_truth

def col_stack(entry_list):
	if len(entry_list[0].shape) > 1:
		for i in xrange(len(entry_list)):
			if i==0:
				container = entry_list[0]
			else:
				container = np.vstack((container, entry_list[i]))
	else:
		for i in xrange(len(entry_list)):
			if i==0:
				container = entry_list[0]
			else:
				container = np.hstack((container, entry_list[i]))
	return container

def shuffle(X,y):
	# Given list1 and list2
	X_train_shuf = []
	y_train_shuf = []
	index_shuf = range(X_train.shape[0])
	np.random.shuffle(index_shuf)
	print "Shuffled"
	for i in index_shuf:
	    X_train_shuf.append(X_train[i,:])
	    y_train_shuf.append(y_train[i])
	print X_train.shape
	print y_train.shape
	print "done"
	return X_train, y_train

# X_train = col_stack(X_train)
# y_train = col_stack(y_train)


splits = 5

def split(X_train, y_train, splits):
	m = X_train.shape[0]
	spacing = int(math.floor(m/splits))
	X_t = []
	y_t = []
	for i in xrange(splits):
		X_t.append(X_train[i::5,:])
		y_t.append(y_train[i::5])
		# print np.unique(y_train[i::5], return_counts = True)
	return X_t, y_t


X_train, y_train = shuffle(X_train, y_train)
X_train, y_train = split(X_train, y_train, splits)
K = 500
n_features = "sqrt"
y_pred = []
y_test = []



print "Data split"
for i in xrange(splits):
	print "Training step: ", i

	X_t_temp = [x for j,x in enumerate(X_train) if j!=i]
	y_t_temp = [x for j,x in enumerate(y_train) if j!=i]	
	rf = randForest(col_stack(X_t_temp), col_stack(y_t_temp), K, n_features = n_features)
	y_pred.append(rf.predict(X_train[i]))
	y_test.append(y_train[i])
	del rf

y_pred = col_stack(y_pred)
y_test = col_stack(y_test)

plotNiceConfusionMatrix(y_test, y_pred, class_names)


code.interact(local=dict(globals(), **locals()))




