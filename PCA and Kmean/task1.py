#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math


# In[10]:


df_train = pd.read_csv('mnist_train.csv',header=None)
X_train = df_train.iloc[:, 1:]
y_train = df_train.iloc[:,0:1]

df_test = pd.read_csv('mnist_test.csv',header=None)
X_test = df_test.iloc[:, 1:]
y_test = df_test.iloc[:,0:1]


# In[11]:


X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)


# In[12]:


def calculate_PCA(X_train, X_test, num_components,):
    X_train_meaned = X_train - np.mean(X_train , axis = 0)
    X_test_meaned = X_test - np.mean(X_train , axis = 0)

    cov_matrix = np.cov(X_train_meaned.T)
    values, vectors = np.linalg.eig(cov_matrix)
    pairs = []
    for i in range(len(values)):
        pairs.append((np.abs(values[i]), vectors[:,i]))
    pairs.sort(key=lambda x: x[0], reverse=True)

    X_train_reduced = []
    X_test_reduced = []
    for i in range(num_components):
        X_train_reduced.append(X_train_meaned.dot(pairs[i][1]))
        X_test_reduced.append(X_test_meaned.dot(pairs[i][1]))
        
    return np.array(X_train_reduced).T, np.array(X_test_reduced).T


# In[13]:


X_train_reduced_1, X_test_reduced_1 = calculate_PCA(X_train, X_test, 4)


# In[15]:


X_train_reduced_1


# In[20]:


pca = PCA (n_components=4)
pca.fit(X_train)
X_train_reduced_2 = pca.transform(X_train)
X_train_reduced_2


# In[ ]:




