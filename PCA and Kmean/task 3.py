#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import math


# In[4]:


df_train = pd.read_csv('mnist_train.csv',header=None)
X_train = df_train.iloc[:, 1:]
y_train = df_train.iloc[:,0:1]

df_test = pd.read_csv('mnist_test.csv',header=None)
X_test = df_test.iloc[:, 1:]
y_test = df_test.iloc[:,0:1]


# In[5]:


X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)


# In[6]:


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
        
    def initialize_centroids(self):
        for i in range(self.num_clusters):
            self.centroids[i] = X_train[np.random.choice(range(self.num_samples))]



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
        #error = distance to the closest centroid
        total_error = 0

        for point in X:
            distances = np.sqrt(np.sum((point - self.centroids)** 2, axis=1))
            closest_centroid = np.argmin(distances)
            total_error += distances[closest_centroid]
            
            return total_error

    def fit(self):

        max_iterations = 100
        self.initialize_centroids()
        errors = []
        for i in range(max_iterations):
            old_centroids = self.centroids
            self.update_clusters()
            new_centroids = self.update_centroids()
            errors.append(self.calculate_error(self.X_train))
              #check convergence
            distance = np.sqrt(np.sum((old_centroids - new_centroids)** 2, axis=1))
            if (np.sum(distance) == 0):
                break
        y_pred = self.add_label()
        self.errors = errors
        self.iterations = i
        
        return y_pred


# In[7]:


clf = KMeanCluster(X_train, 10)
y_pred = clf.fit()
plt.plot(clf.errors)
plt.show()


# In[ ]:




