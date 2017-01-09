from data_tools import *
from algorithms import *
from plot_lib import *
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn import preprocessing
import numpy as np
import code 

# Loading the data into X, y
DATA = loadnumpy("data.npy").astype(np.float64)
# DATA = DATA[:,0:76]
# DATA = DATA[:,76:152]
# DATA = DATA[:,152:228]
# DATA = DATA[:,228:]

# Normalization
for i in xrange(DATA.shape[1]):
	DATA[:,i] = (DATA[:,i] - np.mean(DATA[:,i]))/(np.max(DATA[:,i])-np.min(DATA[:,i])+.001)

ground_truth = loadnumpy("labels.npy").astype(int)
# code.interact(local=dict(globals(), **locals()))
# DATA = preprocessing.scale(DATA)

X_train, X_test, y_train, y_test = train_test_split(DATA, ground_truth, test_size=0.33, random_state=42)


print "Shuffled data and split in train / test"


#Cross Validation
cv = 7


# Random Forest:
# K = 100
# rf = randForest(X_train, y_train, K)
# print "The train accuracy of Random Forest is", np.mean(cross_val_score(rf, X_train, y_train, cv=cv))
# y_pred = rf.predict(X_test)
# plotConfusionMatrix(y_test, y_pred)
# print "The test accuracy is", testAccuracy(y_test.astype(int), y_pred.astype(int))

# T-SNE
X_tSNE = tsne(X_test, 3, 304, 20.0);
plot3d(X_tSNE,y_test,"tSNE-Plot")
code.interact(local=dict(globals(), **locals()))