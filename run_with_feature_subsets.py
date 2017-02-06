from data_tools import *
from algorithms import *
from plot_lib import *
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn import preprocessing
import numpy as np
import code 

# Loading the data into X, y
DATA = loadnumpy("data.npy").astype(np.float64)[:,76:152]
ground_truth = loadnumpy("labels.npy").astype(int)

DATA_ex2 = loadnumpy("ex_2_data_no_zeros.npy").astype(np.float64)[:,76:152]
ground_truth_ex2 = loadnumpy("ex_2_labels_no_zeros.npy").astype(int)

DATA_ex3 = loadnumpy("ex_3_data_no_zeros.npy").astype(np.float64)[:,76:152]
ground_truth_ex3 = loadnumpy("ex_3_labels_no_zeros.npy").astype(int)


class_names = ["0", "1", "2", "3"]

# Omitting single Channels:
# DATA = DATA[:,0:76]
# DATA = DATA[:,76:152]
# DATA = DATA[:,152:228]
# DATA = DATA[:,228:]

# code.interact(local=dict(globals(), **locals()))
# DATA = preprocessing.scale(DATA)



print "Shuffled data and split in train / test"

# X_train, X_test, y_train, y_test = train_test_split(DATA, ground_truth, test_size=0.33, random_state=42)

#Cross Validation
cv = 5


# Random Forest:
K = 500
n_features = "sqrt"

rf = randForest(DATA, ground_truth, K, n_features = n_features)
print "The train accuracy of Random Forest is", np.mean(cross_val_score(rf, DATA, ground_truth, cv=cv))
# y_pred = rf.predict(X_test)
# plotConfusionMatrix(y_test, y_pred)
# print "The test accuracy is", testAccuracy(y_test.astype(int), y_pred.astype(int))
# print "The train accuracy of Random Forest is", np.mean(cross_val_score(rf, DATA, ground_truth, cv=cv))
print "---------------------------------------"
print "Predicting on exp. 2:"
y_pred_ex2 = rf.predict(DATA_ex2)
print "The prediction accuracy on exp. 2 is " + str(accuracy(ground_truth_ex2, y_pred_ex2)) + "%."
print "---------------------------------------"
print "Predicting on exp. 3:"
y_pred_ex3 = rf.predict(DATA_ex3)
print "The prediction accuracy on exp. 3 is " + str(accuracy(ground_truth_ex3, y_pred_ex3)) + "%."



code.interact(local=dict(globals(), **locals()))
plotNiceConfusionMatrix(ground_truth_ex2, y_pred_ex2, class_names)