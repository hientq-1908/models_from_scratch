#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import math


# In[2]:


df_train = pd.read_csv('mnist_train.csv',header=None)
X_train = df_train.iloc[:, 1:]
y_train = df_train.iloc[:,0:1]

df_test = pd.read_csv('mnist_test.csv',header=None)
X_test = df_test.iloc[:, 1:]
y_test = df_test.iloc[:,0:1]


# In[3]:


X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)


# In[4]:


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


# In[5]:


# 1-neares neighbor
# euclidean_distance: np.linalg.norm(row1 - row2)
def predict_1_neighbor(X,y , predicting_point):
    distances = []
    for point in X:
        uclidean_distance = np.linalg.norm(predicting_point - point)
        distances.append((point, uclidean_distance))
        distances.sort(key=lambda x: x[1])
    neighbor = distances[0][0]
    
    for i in range(len(X)):
        if all([j == True for j in np.isclose(neighbor, X[i])]):
            index = i
            break
            
    return y[index]


# In[6]:


scores = []

list_num_components = [10,50,100,150,200,256]
for num in list_num_components:
                    
    X_train_reduced, X_test_reduced = calculate_PCA(X_train, X_test, num)

    y_pred = [predict_1_neighbor(X_train_reduced, y_train, point) for point in X_test_reduced]
    score = accuracy_score(y_pred, y_test)
    scores.append(score)
    print("done:",num, " score", score)


# In[7]:


scores


# In[19]:


plt.plot(scores)
plt.ylim(0.5,1.1)
plt.xlabel('n components')
plt.ylabel('accuracy')
plt.xticks(np.arange(len(list_num_components)),labels=list_num_components)
plt.show()


# In[20]:


def calculate_variances(X_train):
    X_train_meaned = X_train - np.mean(X_train , axis = 0)

    cov_matrix = np.cov(X_train_meaned.T)
    values, vectors = np.linalg.eig(cov_matrix)
    pairs = []
    for i in range(len(values)):
        pairs.append((np.abs(values[i]), vectors[:,i]))
    pairs.sort(key=lambda x: x[0], reverse=True)

    explained_variances = []
    for i in range(len(values)):
        explained_variances.append(pairs[i][0] / np.sum(values))        
    return explained_variances


# In[22]:


variances = calculate_variances(X_train)
totals  = []
arr = np.arange(20)
arr = arr* 10
for i in arr:
    _ = 0
    for j in range(i):
        _ += variances[j]
    totals.append(_)
plt.plot(totals)
plt.xticks(np.arange(20), labels = arr)
plt.show()


# In[ ]:




