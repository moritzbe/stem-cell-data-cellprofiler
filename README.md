#Classification of CellData

The first experiment was performed with some zeros in the cells that slightly increased the average intensities for the thresholds. In the second experiment, these zeros are encountered for with no or little changes to the results.

Logistic Regression:
- reg = .33, train_error = 0.944539648041

Random Forest:
- K = 30, train_error = 0.96160875858
- K = 100, train_error = 0.962, Test Error is: 96%
[[  0.00000000e+00   0.00000000e+00   1.00000000e+00   2.00000000e+00
    3.00000000e+00]
 [  0.00000000e+00   1.01000000e+03   9.00000000e+00   5.10000000e+01
    2.00000000e+00]
 [  1.00000000e+00   2.00000000e+01   4.29000000e+02   0.00000000e+00
    2.50000000e+01]
 [  2.00000000e+00   4.90000000e+01   0.00000000e+00   2.66400000e+03
    3.00000000e+01]
 [  3.00000000e+00   5.00000000e+00   4.30000000e+01   7.90000000e+01
    3.57800000e+03]]
- PCA and Random Forest, K = 30, train_error = 0.85


Using Random Forest for 1 Color Channels (K=30): 
- Channel 1: accuracy = 54%
[[  0.00000000e+00   0.00000000e+00   1.00000000e+00   2.00000000e+00
    3.00000000e+00]
 [  0.00000000e+00   1.90000000e+02   4.00000000e+00   3.84000000e+02
    4.94000000e+02]
 [  1.00000000e+00   3.50000000e+01   8.00000000e+01   1.08000000e+02
    2.51000000e+02]
 [  2.00000000e+00   1.28000000e+02   8.00000000e+00   1.66600000e+03
    9.41000000e+02]
 [  3.00000000e+00   1.75000000e+02   6.10000000e+01   1.10000000e+03
    2.36900000e+03]]
- Channel 2: accuracy = 77%
[[  0.00000000e+00   0.00000000e+00   1.00000000e+00   2.00000000e+00
    3.00000000e+00]
 [  0.00000000e+00   9.53000000e+02   5.80000000e+01   4.20000000e+01
    1.90000000e+01]
 [  1.00000000e+00   2.59000000e+02   1.65000000e+02   2.90000000e+01
    2.10000000e+01]
 [  2.00000000e+00   4.30000000e+01   1.60000000e+01   2.11300000e+03
    5.71000000e+02]
 [  3.00000000e+00   2.20000000e+01   1.80000000e+01   7.90000000e+02
    2.87500000e+03]]
- Channel 3: accuracy = 81%
[[  0.00000000e+00   0.00000000e+00   1.00000000e+00   2.00000000e+00
    3.00000000e+00]
 [  0.00000000e+00   4.48000000e+02   1.00000000e+00   6.14000000e+02
    9.00000000e+00]
 [  1.00000000e+00   3.00000000e+00   9.00000000e+01   1.60000000e+01
    3.65000000e+02]
 [  2.00000000e+00   3.26000000e+02   1.00000000e+00   2.37100000e+03
    4.50000000e+01]
 [  3.00000000e+00   6.00000000e+00   6.30000000e+01   6.40000000e+01
    3.57200000e+03]]
- Channel 4: accuracy = 50%
[[  0.00000000e+00   0.00000000e+00   1.00000000e+00   2.00000000e+00
    3.00000000e+00]
 [  0.00000000e+00   1.09000000e+02   1.00000000e+00   4.15000000e+02
    5.47000000e+02]
 [  1.00000000e+00   3.40000000e+01   6.00000000e+00   1.13000000e+02
    3.21000000e+02]
 [  2.00000000e+00   1.00000000e+02   3.00000000e+00   1.57500000e+03
    1.06500000e+03]
 [  3.00000000e+00   1.40000000e+02   6.00000000e+00   1.08200000e+03
    2.47700000e+03]]


SVM and PCA
- 94.6%



Experiment 2:
Unique labels are: [ 0.  1.  2.  3.  4.]
Label Frequencies are: [ 4137  1833 11191 14506  1196]

Experiment 3:
Unique labels are: [ 0.  1.  2.  3.  4.]
Label Frequencies are: [ 5775  2162 12133 14021  1225]



ToDos:
- check pipeline
- check dataextraction (any labels in the data?)
- normalization
- find the most discerning features

## Inception using AlexNet on Ch1,Ch2,Ch3, Predicting 5 classes (not 4 like with cellprofiler)
- Random Forest: Normalization, PCA reducing feature dimesnions to 100, K=30 produces 73% accuracy.
[[  0.00000000e+00   0.00000000e+00   1.00000000e+00   2.00000000e+00
    3.00000000e+00   4.00000000e+00]
 [  0.00000000e+00   6.87000000e+02   1.00000000e+00   3.15000000e+02
    6.10000000e+01   2.00000000e+00]
 [  1.00000000e+00   3.80000000e+01   9.40000000e+01   3.10000000e+01
    2.78000000e+02   3.40000000e+01]
 [  2.00000000e+00   2.76000000e+02   2.00000000e+00   2.21300000e+03
    2.32000000e+02   5.00000000e+00]
 [  3.00000000e+00   7.10000000e+01   3.10000000e+01   4.85000000e+02
    3.05300000e+03   5.60000000e+01]
 [  4.00000000e+00   1.20000000e+01   1.00000000e+01   5.20000000e+01
    2.09000000e+02   1.68000000e+02]]

- Logistic Regression, Normalization, PCA reducing feature dimesnions to 300, reg=1 produces 79% accuracy. Removing the last class gives 82% accuracy.


## Experiments, cv=5:
# 1-st pretest 
17.01.17:
K = 30, n_features = 30
96% accuracy on Train and Testset

# 1st 
17.01.17:
K = 500, n_features = "auto"
96.4% accuracy on Train and 96.7% on Testset

# 2nd 
17.01.17:
K = 500, n_features = 10
96.0% accuracy on Train and 96.3% on Testset

# 3rd 
17.01.17:
K = 500, n_features = 200
96.8% accuracy on Train and 97.1% on Testset

