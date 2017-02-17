from data_tools import *
from algorithms import *
from plot_lib import *
# from nets import *
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import numpy as np
import code 


# Loading the data into X, y
DATA = loadnumpy("data_no_zeros.npy").astype(np.float64)
ground_truth = loadnumpy("labels_no_zeros.npy").astype(int)

DATA_ex2 = loadnumpy("ex_2_data_no_zeros.npy").astype(np.float64)
ground_truth_ex2 = loadnumpy("ex_2_labels_no_zeros.npy").astype(int)

DATA_ex3 = loadnumpy("ex_3_data_no_zeros.npy").astype(np.float64)
ground_truth_ex3 = loadnumpy("ex_3_labels_no_zeros.npy").astype(int)

m_total = DATA.shape[0] + DATA_ex2.shape[0] + DATA_ex3.shape[0]

print "Loaded data and ground_truth of exp 1, 2 and 3.."

# Normalization
# DATA = normalize(DATA)
# print "Normalized Data."
data_entries = [[DATA,ground_truth],[DATA_ex2,ground_truth_ex2],[DATA_ex3,ground_truth_ex3]]
channel_importances = []
feature_importances = []
for entries in data_entries:
	X_train, X_test, y_train, y_test = train_test_split(entries[0], entries[1], test_size=None, random_state=42)
	print "Shuffled data and split in train-/testset."
	# Feature importance selection
	K = 500
	# clf = ExtraTreesClassifier(n_estimators = K, max_features = "sqrt", verbose = 1)
	clf = RandomForestClassifier(n_estimators = K, max_features = "sqrt", verbose = 1)
	clf = clf.fit(X_train, y_train)
	print "Feature importances are:", clf.feature_importances_
	print "Mean feature importance is:", np.mean(clf.feature_importances_)
	print "Plotting histogram:"
	# plotHistogram(clf.feature_importances_, bins=30, xlabel="Importance-Score", ylabel="Feature-Frequency", title="Feature Importances")
	print "The maximum feature importance is:", np.max(clf.feature_importances_)
	n_max_features = 10
	print "The indices of the "+ str(n_max_features)+ " most important features are:"
	print clf.feature_importances_.argsort()[-n_max_features:][::-1]
	print "(listed in order of magnitude)"
	print "--------------------------------"
	print "Importance of the single channels by summing importance scores:"
	# Indices alsways run [start, finish] - excluding finish!
	ch1 = np.sum(clf.feature_importances_[0:76])/np.sum(clf.feature_importances_)*100
	ch2 = np.sum(clf.feature_importances_[76:152])/np.sum(clf.feature_importances_)*100
	ch3 = np.sum(clf.feature_importances_[152:228])/np.sum(clf.feature_importances_)*100
 	ch4 = np.sum(clf.feature_importances_[228:304])/np.sum(clf.feature_importances_)*100
	print "Ch1: " + str(ch1)+"%."
	print "Ch2: " + str(ch2)+"%."
	print "Ch3: " + str(ch3)+"%."
	print "Ch4: " + str(ch4)+"%."
	channel_importances += [ch1 * entries[0].shape[0],ch2 * entries[0].shape[0],ch3 * entries[0].shape[0],ch4 * entries[0].shape[0]]
	feature_importances += [clf.feature_importances_ * entries[0].shape[0]]

total_importances = (feature_importances[0]+feature_importances[1]+feature_importances[2])/m_total
print total_importances[total_importances.argsort()[-n_max_features:][::-1]]


code.interact(local=dict(globals(), **locals()))
least = total_importances.argsort()[n_max_features:][::-1]
most = total_importances.argsort()[-n_max_features:][::-1]
# calculate correlation between features
# take most important, set max_features None
# Lasso


X_train = clf.transform(X_train)
X_test = clf.transform(X_test)

print "Feature selection done"