import pandas as pd
import numpy as np
import code

path_to_data = '/Volumes/MoritzBertholdHD/CellData/featureExtractionCellProfiler/extracted_data/'

ch1 = pd.DataFrame.from_csv(path_to_data + 'data_ch1_no_zeros.csv', header=0, sep=',', index_col=None)
ch2 = pd.DataFrame.from_csv(path_to_data + 'data_ch2_no_zeros.csv', header=0, sep=',', index_col=None)
ch3 = pd.DataFrame.from_csv(path_to_data + 'data_ch3_no_zeros.csv', header=0, sep=',', index_col=None)
ch4 = pd.DataFrame.from_csv(path_to_data + 'data_ch4_no_zeros.csv', header=0, sep=',', index_col=None)

# +1 to the features you want to drop
drop_cols_labels = [0, 1, 11, 22, 23, 24, 25, 26, 27, 28, 29]

ch1 = ch1[pd.notnull(ch1['image_filenames'])]
y = ch1['class_labels']
names = ch1['image_filenames']
ch1.drop(ch1.columns[drop_cols_labels],axis=1,inplace=True)
ch2 = ch2[pd.notnull(ch2['image_filenames'])]
ch2.drop(ch2.columns[drop_cols_labels],axis=1,inplace=True)
ch3 = ch3[pd.notnull(ch3['image_filenames'])]
ch3.drop(ch3.columns[drop_cols_labels],axis=1,inplace=True)
ch4 = ch4[pd.notnull(ch4['image_filenames'])]
ch4.drop(ch4.columns[drop_cols_labels],axis=1,inplace=True)


data = pd.concat([ch1, ch2, ch3, ch4], axis=1)

# To numpy
data_numpy = data.iloc[:,:].values
y_numpy = y.iloc[:].values

np.save('data_no_zeros.npy', data_numpy, allow_pickle=True, fix_imports=True)
np.save('labels_no_zeros.npy', y_numpy, allow_pickle=True, fix_imports=True)



# code.interact(local=dict(globals(), **locals()))


# access entries:
# ch1.ix[0,0]


# Findings: 
# data_21 - data_28, and data_10 is empty in all channels
# There are IDs: "NaN" in the data