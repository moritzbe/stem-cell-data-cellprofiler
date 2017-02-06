
# coding: utf-8

# In[64]:

import pandas as pd
data = pd.read_csv('SSC.txt', delim_whitespace = True, header=0)


# In[65]:

data.shape


# In[66]:

data.head(1)


# In[67]:

a = range(1,50)
b = range(71, 77)
a.extend(b)
drop_cols_labels = [0, 1, 11, 22, 23, 24, 25, 26, 27, 28, 29]
print a


# In[68]:

data.drop(data.columns[a],axis=1,inplace=True) # leaves 85
data.drop(data.columns[drop_cols_labels],axis=1,inplace=True) # leaves 74


# In[69]:

data.shape


# In[88]:

label_list = []
label_list.extend(data.columns.values)
label_list.extend(data.columns.values)
label_list.extend(data.columns.values)
label_list.extend(data.columns.values)


# In[94]:

print len(label_list)
print "______________________________"
print "Most important features Channel 2"
for i in [86,  93,  82,  88,  94]:
    print label_list[i]
print "______________________________"
print "Most important features Channel 3"
for i in [169, 164, 158, 162, 170]:
    print label_list[i]


# In[ ]:



