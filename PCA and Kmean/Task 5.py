#!/usr/bin/env python
# coding: utf-8

# In[102]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import svm
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
import math


# In[39]:


df_train = pd.read_csv('mnist_train.csv',header=None)
X_train = df_train.iloc[:, 1:]
y_train = df_train.iloc[:,0:1]

df_test = pd.read_csv('mnist_test.csv',header=None)
X_test = df_test.iloc[:, 1:]
y_test = df_test.iloc[:,0:1]


# In[40]:


X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)


# In[41]:


num_clusters = 10
num_samples, num_features = X_train.shape


# In[42]:


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


# In[45]:


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


# In[43]:


noise_X_train = np.random.normal(loc=0.0, scale=1.0, size=(len(X_train),256))
noise_X_test = np.random.normal(loc=0.0, scale=1.0, size=(len(X_test),256))


# In[44]:


X_train_noised = np.append(X_train, noise_X_train, axis = 1)
X_test_noised = np.append(X_test, noise_X_test, axis = 1)


# # I. Performance without using PCA and no noise

# ## I.1 1-nearest neighbor

# In[ ]:


y_pred = [predict_1_neighbor(X_train, y_train, point) for point in X_test]


# In[51]:


score = accuracy_score(y_pred, y_test)
score


# ## I.2 SVC

# In[61]:


clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X_train,y_train.ravel())


# In[62]:


score_svc_original = accuracy_score(clf.predict(X_test), y_test)
score_svc_original


# ## II. Performance with noised dataset and without PCA

# ## II.1 1-nearest neighbor 

# In[63]:


y_pred = [predict_1_neighbor(X_train_noised, y_train, point) for point in X_test_noised]


# In[65]:


score_1_noised = accuracy_score(y_pred, y_test)
score_1_noised


# ## II.2 SVC

# In[67]:


clf_noised = svm.SVC(kernel='linear', C = 1.0)
clf_noised.fit(X_train_noised,y_train.ravel())


# In[92]:


score_svc_noised = accuracy_score(clf_noised.predict(X_test_noised), y_test)
score_svc_noised


# ## III. Performance with noised data and PCA

# In[89]:


X_train_reduced, X_test_reduced = calculate_PCA (X_train_noised, X_test_noised, 50)


# ## III.1 1-nearest neighbor

# In[95]:


y_pred = [predict_1_neighbor(X_train_reduced, y_train, point) for point in X_test_reduced]


# In[101]:


score_1_noised_pca = accuracy_score(y_pred, y_test)
score_1_noised_pca


# ## III.2 SVC

# In[99]:


clf_noised_pca = svm.SVC(kernel='linear', C = 1.0)
clf_noised_pca.fit(X_train_reduced.real,y_train.ravel())


# In[100]:


score_svc_noised_pca = accuracy_score(clf_noised_pca.predict(X_test_reduced.real), y_test)
score_svc_noised_pca


# In[ ]:





# In[104]:


def calculate_distribution(X_train):
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


# In[114]:


variances = calculate_distribution(X_train)
plt.plot(variances[0:100])
plt.title("Without noise")
plt.show()


# In[113]:


variances = calculate_distribution(X_train_noised)
plt.plot(variances[0:100])
plt.title("Noise")
plt.show()


# In[ ]:




