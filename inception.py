# coding: utf-8

%matplotlib inline

from data_tools import *
from algorithms import *
from plot_lib import *
import matplotlib.pyplot as plt
# from nets import *
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn import preprocessing
import numpy as np
import code 
import caffe


# Loading the data into X, y

print "Loaded data and ground_truth of exp 1."

net = caffe.Net('ModelZoo/conv.prototxt', caffe.TEST)
print "Loaded the net"
code.interact(local=dict(globals(), **locals()))
