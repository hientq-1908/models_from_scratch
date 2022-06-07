#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt
import pandas as pd
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


def svm_train_primal( data_train , label_train , regularisation_para_C):
    
    X_train = data_train
    y_train = label_train
    C = regularisation_para_C

    #number of features
    m = len(X_train[0])

    #number of samples
    n = len(y_train)

    P = np.zeros((m+n+1,m+n+1), float)
    
    #create diagonal for H
    diag_P = [0]
    for i in range(m):
        diag_P.append(1)
    for i in range(n):
        diag_P.append(0)

    np.fill_diagonal(P,diag_P)

    q = np.r_[np.zeros(1+m), np.ones(n) * C]

    X_y = []
    for i in range(n):
        for j in range(m):
            X_y.append(X_train[i][j] * y_train[i])
    X_y = np.array(X_y).reshape(n,-1)

    G = np.concatenate((y_train.reshape(-1,1),X_y, np.eye(n)* -1), axis = 1)
    G_bottom = np.concatenate((np.zeros((n,m+1), float), np.eye(n) * -1), axis = 1)

    G = np.concatenate((G, G_bottom), axis = 0)

    b = np.zeros(1)
    A = np.zeros((1,m+n+1),float)
    h = np.concatenate((np.ones((n,1), float)*-1,np.zeros((n,1), float)), axis =0)

    P = cvxopt_matrix(P)
    G = cvxopt_matrix(G)
    q = cvxopt_matrix(q)
    h = cvxopt_matrix(h)


    sol = cvxopt_solvers.qp(P, q, G, h)
    v = np.array(sol['x'])
    v = v.T[0]
    w0 = v[0]
    w = v[1:len(v)-n]
    
    return [-w,-w0]


# In[5]:


def svm_predict_primal(data_test , label_test , svm_model):
    X_test = data_test
    y_test = label_test
    w0 = svm_model[1]
    w = svm_model[0]
    
    y_pred = []
    for x in X_test:
        f = w.T @ x + w0
        if f > 0 :
            y_pred.append(1)
        else:
            y_pred.append(-1)
    
    return accuracy_score(y_test, y_pred)    


# In[6]:


svm_model = svm_train_primal( X_train , y_train , 60)


# In[12]:


print('w0',svm_model[1])
print('w', svm_model[0])


# In[8]:


test_accuracy = svm_predict_primal(X_train , y_train , svm_model)
print("accucracy on training set",test_accuracy)
test_accuracy_2 = svm_predict_primal(X_test , y_test, svm_model)
print("accuracy on testing set", test_accuracy_2)


# In[6]:


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
        svm_model = svm_train_primal(X_train_sub,y_train_sub, C)
        test_accuracy_d = svm_predict_primal(X_test_sub,y_test_sub,svm_model)
        accuracy_scores.append(test_accuracy_d)


# In[7]:


scores = np.array(accuracy_scores)
scores = scores.reshape(-1,5)
#calculating average accuracy scores of C= 30, 60, 90
avg_scores = np.mean(scores, axis = 1)
print(avg_scores)


# In[ ]:




