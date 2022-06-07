#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


# In[4]:


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


# In[5]:


X_train = np.array(df_X_train)
y_train = np.array(df_y_train)
X_test = np.array(df_X_test)
y_test = np.array(df_y_test)


# In[6]:


def svm_train_dual(data_train , label_train , regularisation_para_C):
    X_train = data_train
    y_train = label_train
    C = regularisation_para_C
    
    #number of samples
    N = len(y_train)

    # H = yi * yj * Xi * Xj = (y*X)T  * (y*X)
    y_train = y_train.reshape(-1,1)
    X_ = y_train * X_train
    H = X_ @ X_.T

    #initializing cvxopt parameters
    G = np.r_[(np.eye(N)*-1,np.eye(N))]
    q = np.ones(N) * -1
    h = np.r_[np.zeros(N), np.ones(N) * C]
    b = np.zeros(1)
    A = y_train.reshape(1,-1)

    P = cvxopt_matrix(H)
    G = cvxopt_matrix(G)
    q = cvxopt_matrix(q)
    h = cvxopt_matrix(h)
    b = cvxopt_matrix(b)
    A = cvxopt_matrix(A)
    
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])


    # w = sum(alpha * y * x) 
    w = 0
    for i in range(N):
        w += alphas[i] * y_train[i] * X_train[i]
    
    
    # w0 = 1/yi - wTxi
    # because of wide range of w0s, we will get the mean
    epsilon = 1e-6
    w0 = []
    for i in range(N):
        if (alphas[i] > epsilon):
            w0.append(1/y_train[i] - w.T @ X_train[i])

    w0 = np.array(w0)
    w0 = w0.mean()
    
    return [w, w0]


# In[7]:


def svm_predict_dual(data_test , label_test , svm_model_d):
    X_test = data_test
    y_test = label_test
    w0 = svm_model_d[1]
    w = svm_model_d[0]
    
    y_pred = []
    for x in X_test:
        f = w.T @ x + w0
        if f > 0 :
            y_pred.append(1)
        else:
            y_pred.append(-1)
    
    return accuracy_score(y_test, y_pred)


# In[8]:


svm_model_d = svm_train_dual(X_train , y_train , 60 )


# In[13]:


print('w0',svm_model_d[1])
print('w',svm_model_d[0])


# In[9]:


test_accuracy_d = svm_predict_dual(X_train , y_train , svm_model_d)
print("accucracy on training set",test_accuracy_d)
test_accuracy_d_2 = svm_predict_dual(X_test , y_test, svm_model_d)
print("accuracy on testing set", test_accuracy_d_2)


# In[12]:


#tuning C by k-fold cross validation
Cs = [30, 60, 90]

#Kfold with k = 5
kf = KFold(n_splits=5,shuffle=False)
kf.split(df_X_train)

accuracy_scores = []


for C in Cs:
    print("Calculating average accuracy score with C =", C)
    for train_index, test_index in kf.split(df_X_train):
        df_X_train_sub, df_X_test_sub = df_X_train.iloc[train_index], df_X_train.iloc[test_index]
        df_y_train_sub, df_y_test_sub = df_y_train[train_index], df_y_train[test_index]

        X_train_sub = np.array(df_X_train_sub)
        X_test_sub = np.array(df_X_test_sub)
        y_train_sub = np.array(df_y_train_sub)
        y_test_sub = np.array(df_y_test_sub)
        svm_model_d = svm_train_dual(X_train_sub,y_train_sub, C)
        test_accuracy_d = svm_predict_dual(X_test_sub,y_test_sub,svm_model_d)
        accuracy_scores.append(test_accuracy_d)


# In[22]:


scores = np.array(accuracy_scores)
scores = scores.reshape(-1,5)
#calculating average accuracy scores of C= 30, 60, 90
avg_scores = np.mean(scores, axis = 1)
print(avg_scores)


# In[ ]:




