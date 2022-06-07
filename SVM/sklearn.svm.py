#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


# In[2]:


#loading dataset
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#selecting label and replacing 0 by -1
df_y_train = df_train.iloc[:,0]
df_y_train[df_y_train==0]=-1
df_y_test = df_test.iloc[:,0]
df_y_test[df_y_test==0]=-1

#selecting features
df_X_train = df_train.iloc[:,1:]
df_X_test = df_test.iloc[:,1:]


# In[3]:


X_train = np.array(df_X_train)
y_train = np.array(df_y_train)
X_test = np.array(df_X_test)
y_test = np.array(df_y_test)


# In[4]:


clf = SVC(C = 60, kernel = 'linear')
clf.fit(X_train, y_train.ravel())


# In[10]:


w = clf.coef_[0]
w0 = clf.intercept_


# In[14]:


print(w0)
print(w)


# In[11]:


def svm_predict_sklearn(data_test , label_test , w, w0):
    X_test = data_test
    y_test = label_test
    
    y_pred = []
    for x in X_test:
        f = w.T @ x + w0
        if f > 0 :
            y_pred.append(1)
        else:
            y_pred.append(-1)
    
    return accuracy_score(y_test, y_pred) 


# In[12]:


test_accuracy = svm_predict_sklearn(X_train , y_train , w, w0)
print("accucracy on training set",test_accuracy)
test_accuracy_2 = svm_predict_sklearn(X_test , y_test, w, w0)
print("accuracy on testing set", test_accuracy_2)


# In[ ]:




