import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D




 
objects = ('Ch1: PGP only', 'Ch2: PGP+CGRP', 'Ch3: PGP+RIIb', 'Ch4: DAPI')
y_pos = np.arange(len(objects))
importance = [11.38,32.35,50.00,6.28]
 
plt.bar(y_pos, importance, align='center', alpha=0.5)
plt.ylim([0, 60]) 
plt.xticks(y_pos, objects)
plt.ylabel('Score')
plt.title('Channel Importance')
 
plt.show()