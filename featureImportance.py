from data_tools import *
from algorithms import *
from plot_lib import *
# from nets import *
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
import numpy as np
import code 


# Loading the data into X, y
DATA = loadnumpy("data_no_zeros.npy").astype(np.float64)
ground_truth = loadnumpy("labels_no_zeros.npy").astype(int)
print "Loaded data and ground_truth of exp 1."

# Normalization
# DATA = normalize(DATA)
# print "Normalized Data."

X_train, X_test, y_train, y_test = train_test_split(DATA, ground_truth, test_size=None, random_state=42)
print "Shuffled data and split in train-/testset."


# Feature importance selection
K = 30
clf = ExtraTreesClassifier(n_estimators = K, max_features = None, verbose = 1)
clf = clf.fit(X_train, y_train)
print "Feature importances are:", clf.feature_importances_
print "Mean feature importance is:", np.mean(clf.feature_importances_)
print "Plotting histogram:"
plotHistogram(clf.feature_importances_, bins=30, xlabel="Importance-Score", ylabel="Feature-Frequency", title="Feature Importances")
print "The maximum feature importance is:", np.max(clf.feature_importances_)
n_max_features = 10
print "The indices of the "+ str(n_max_features)+ " most important features are:"
print clf.feature_importances_.argsort()[-n_max_features:][::-1]
print "(listed in order of magnitude)"
print "--------------------------------"
print "Importance of the single channels by summing importance scores:"
print "Ch1: " + str((np.sum(clf.feature_importances_[0:76])/np.sum(clf.feature_importances_))*100)+"%."
print "Ch2: " + str((np.sum(clf.feature_importances_[76:152])/np.sum(clf.feature_importances_))*100)+"%."
print "Ch3: " + str((np.sum(clf.feature_importances_[152:228])/np.sum(clf.feature_importances_))*100)+"%."
print "Ch4: " + str((np.sum(clf.feature_importances_[228:304])/np.sum(clf.feature_importances_))*100)+"%."

code.interact(local=dict(globals(), **locals()))


X_train = clf.transform(X_train)
X_test = clf.transform(X_test)


print "Feature selection done"

