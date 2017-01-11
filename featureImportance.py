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

X_train, X_test, y_train, y_test = train_test_split(DATA, ground_truth, test_size=0.33, random_state=42)

# Normalization
for i in xrange(DATA.shape[1]):
	DATA[:,i] = (DATA[:,i] - np.mean(DATA[:,i]))/(np.max(DATA[:,i])-np.min(DATA[:,i])+.001)


print "Shuffled data and split in train-/testset."

clf = ExtraTreesClassifier(n_estimators=100, max_features=10, verbose = 1)
clf = clf.fit(X_train, y_train)
print clf.feature_importances_
print "Number of features considered:", clf.n_features_
print "Number of outputs:", clf.n_outputs_


X_train = clf.transform(X_train)
X_test = clf.transform(X_test)


print "Feature selection done"

# Lasso
# alpha = .05 #if alpha = 0, use normal unreg.
# tol = 0.000001
# lasso = lasso(X_train, y_train, alpha, tol)
# print "Lasso's sparse coefficients are:", lasso.sparse_coef_ 
# c1 = 20
# log_regl = logRegress(X_off[:,[0, 3]], y, c1)
# print "With the selected Coefficients, Logistic Regression gives accuracy:", np.mean(cross_val_score(log_regl, X_off[:,[0, 3]], y, cv=cv))
# plotConfusionMatrix(y, log_regl.predict(X_off[:,[0, 3]]))


#Cross Validation
cv = 7

# Random Forest:
K = 200
rf = randForest(X_train, y_train, K)
print "The train accuracy of Random Forest is", np.mean(cross_val_score(rf, X_train, y_train, cv=cv))
y_pred = rf.predict(X_test)
plotConfusionMatrix(y_test, y_pred)

code.interact(local=dict(globals(), **locals()))
