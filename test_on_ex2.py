from data_tools import *
from algorithms import *
from plot_lib import *
# from nets import *
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn import preprocessing
import numpy as np
import code 


# Loading the data into X, y
DATA = loadnumpy("data_no_zeros.npy").astype(np.float64)
ground_truth = loadnumpy("labels_no_zeros.npy").astype(int)

DATA_ex2 = loadnumpy("ex_2_data_no_zeros.npy").astype(np.float64)
ground_truth_ex2 = loadnumpy("ex_2_labels_no_zeros.npy").astype(int)

class_names = ["Class 0", "Class 1", "Class 2", "Class 3"]


print "Loaded data and ground_truth of exp 1 and 2."

# Normalization
for i in xrange(DATA.shape[1]):
	DATA[:,i] = (DATA[:,i] - np.mean(DATA[:,i]))/(np.max(DATA[:,i])-np.min(DATA[:,i])+.001)

for i in xrange(DATA_ex2.shape[1]):
	DATA_ex2[:,i] = (DATA_ex2[:,i] - np.mean(DATA_ex2[:,i]))/(np.max(DATA_ex2[:,i])-np.min(DATA_ex2[:,i])+.001)

print "The test data will be the data of exp 2."
X_train = DATA
y_train = ground_truth

# Perform PCA: 
# th = 5
# pca = PCA()
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# print "PCA done"
# plot3d(X_train[::th,:], ground_truth[::th], "PCA")


#Cross Validation
cv = 7

# Random Forest:
print "Training on exp. 1:"
K = 50
rf = randForest(X_train, y_train, K)
print "The train accuracy of Random Forest is", np.mean(cross_val_score(rf, X_train, y_train, cv=cv))
print "Predicting on exp. 2:"
y_pred_ex2 = rf.predict(DATA_ex2)
plotNiceConfusionMatrix(ground_truth_ex2, y_pred_ex2, class_names)
print "The prediction accuracy is on exp. 2 is " + str(accuracy(ground_truth_ex2, y_pred_ex2)) + "%."

