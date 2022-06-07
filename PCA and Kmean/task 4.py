#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import math


# In[34]:


df_train = pd.read_csv('mnist_train.csv',header=None)
X_train = df_train.iloc[:, 1:]
y_train = df_train.iloc[:,0:1]

df_test = pd.read_csv('mnist_test.csv',header=None)
X_test = df_test.iloc[:, 1:]
y_test = df_test.iloc[:,0:1]


# In[35]:


X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)


# In[36]:


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
        
    return np.array(X_train_reduced).T.real, np.array(X_test_reduced).T.real


# In[37]:


# Generate initial hard centroids from index

def get_initial_hard_centroids(X_train, num_clusters, num_features, hard_centroids_index):
    hard_centroids = np.zeros((num_clusters, num_features))

    for i in range(num_clusters):
        hard_centroids[i] = X_train[hard_centroids_index[i]]
        
    return hard_centroids


# In[38]:


class KMeanCluster():
    def __init__(self, X_train, num_clusters):
        self.num_clusters = num_clusters
        self.X_train = X_train
        self.X_test = X_test
        self.num_samples, self.num_features = X_train.shape
        self.centroids = np.zeros((self.num_clusters, self.num_features))
        self.clusters =  [[] for i in range(self.num_clusters)]
        self.errors = []
        self.iterations = -1
        
    def initialize_centroids(self, initial_centroids):
        self.centroids = initial_centroids



    def update_clusters(self):
        new_clusters =  [[] for i in range(self.num_clusters)]

        for index, point in enumerate(self.X_train):
            distances = np.sqrt(np.sum((point - self.centroids)** 2, axis=1))
            closest_centroid = np.argmin(distances)
            
            new_clusters[closest_centroid].append(index)
    
        self.clusters = new_clusters

        return new_clusters

    def update_centroids(self):
        new_centroids = np.zeros((self.num_clusters, self.num_features))

        for index, cluster in enumerate(self.clusters):
            new_centroid = np.mean(self.X_train[cluster], axis = 0)
            new_centroids[index] = new_centroid
            
            self.centroids = new_centroids

        return new_centroids


    def add_label(self):
        y_pred = np.zeros(self.num_samples)

        for index_cluster, cluster in enumerate(self.clusters):
            for index_sample in cluster:
                y_pred[index_sample] = index_cluster
        return y_pred

    def calculate_error(self, X):
        #error = distance to the closet centroid
        total_error = 0

        for point in X:
            distances = np.sqrt(np.sum((point - self.centroids)** 2, axis=1))
            closest_centroid = np.argmin(distances)
            total_error += distances[closest_centroid]
            
            return total_error

    def fit(self, initial_centroids):

        max_iterations = 100
        self.initialize_centroids(initial_centroids)
        errors = []
        for i in range(max_iterations):
            old_centroids = self.centroids
            self.update_clusters()
            new_centroids = self.update_centroids()
            errors.append(self.calculate_error(self.X_train))
              #check coverage
            distance = np.sqrt(np.sum((old_centroids - new_centroids)** 2, axis=1))
            if (np.sum(distance) == 0):
                break
        y_pred = self.add_label()
        self.errors = errors
        self.iterations = i
        
        return y_pred


# In[ ]:


def calcuate_accuracy(cluster, num_cluster, y_train):
    # checking correct points in each cluster from 0-9
    arr = []
    for i in range(num_cluster):
        cnt = 0
        for j in clf.clusters[i]:
            if y_train[j] == i:
                cnt +=1
        arr.append(cnt)
    # number of points in each cluster from 0-9
    arr2 = []
    for i in clf.clusters:
        arr2.append(len(i))
    
    #calculate accuracy and average
    arr = np.array(arr)
    arr2 = np.array(arr2)
    pct = np.zeros(num_cluster)
    for i in range(len(arr)):
        pct[i] = arr[i]/arr2[i]
    return np.sum(pct)/num_cluster


# In[39]:


num_clusters = 10
num_samples, num_features = X_train.shape


# In[45]:


hard_centroids_index = []

#Generate hard centroid indexes from 10 classes randomlly
for i in range(num_clusters):
    _ = np.where(y_train == i)
    hard_centroids_index.append(_[0][np.random.choice(range(len(_)))])


    
hard_centroids = get_initial_hard_centroids(X_train, num_clusters, num_features, hard_centroids_index)


# In[46]:


clf = KMeanCluster(X_train, 10)
y_pred = clf.fit(hard_centroids)
calcuate_accuracy(clf.clusters, num_clusters, y_train)


# In[47]:


list_num_components = [10,30,90,130,160,200,230,256]      
scores = []
for num in list_num_components:
    X_train_reduced, X_test_reduced = calculate_PCA(X_train, X_test, num)
    hard_centroids = get_initial_hard_centroids(X_train_reduced, num_clusters,num,hard_centroids_index)    
    clf = KMeanCluster(X_train_reduced, 10)
    y_pred = clf.fit(hard_centroids)
    
    score = calcuate_accuracy(clf.clusters, num_clusters, y_train)
    scores.append(score)


# In[48]:


plt.plot(scores)
plt.ylim(0,1)
plt.xticks(np.arange(len(list_num_components)), labels = list_num_components)
plt.xlabel("n_components")
plt.ylabel("accuracy score")

