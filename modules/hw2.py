import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.special as sp
import time
from scipy.optimize import minimize

import data_generator as dg

# you can define/use whatever functions to implememt

########################################
# Part 1. cross entropy loss
########################################
def cross_entropy_softmax_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x = np.reshape(x.T, (-1, n))
    # this will give you a score matrix s of size (num_class)-by-(n)
    # the i-th column vector of s will be
    # the score vector of size (num_class)-by-1, for the i-th input data point
    # performing s=Wx+b
    s = np.dot(W,x) + b
    #calculate softmax
    softmax_initial = np.exp(s)
    softmax = softmax_initial / np.sum(softmax_initial,axis=0)

    #calculate one hot encoding
    num = np.unique(y, axis=0)
    num = num.shape[0]
    one_hot = np.eye(num)[y]

    #calculate cross_entropy_loss
    cross_loss = one_hot.T * np.log(softmax)
    cross_loss = -np.sum(cross_loss)
    # return cross entropy loss
    return cross_loss/n

########################################
# Part 2. SVM loss calculation
########################################
def svm_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x = np.reshape(x.T, (-1, n))
    # this will give you a score matrix s of size (num_class)-by-(n)
    # the i-th column vector of s will be
    # the score vector of size (num_class)-by-1, for the i-th input data point
    # performing s=Wx+b
    s = np.dot(W,x) + b
    sum_loss = 0
    for i in range(n):
        y_tmp = y[i]
        for j in range(0, num_class):
            if j == i:
                continue
            else:
                tmp = s[j,i] - s[y_tmp,i] + 1
                sum_loss = sum_loss + max(tmp, 0)
    avg_loss = sum_loss / n
    # return SVM loss
    return avg_loss

########################################
# Part 3. kNN classification
########################################
def knn_test(X_train, y_train, X_test, y_test, n_train_sample, n_test_sample, k):

    # implement your function here
    dists = compute_distances(X_train, X_test)
    indices = np.argsort(dists, axis=1)
    indices = indices[:,:k]
    tmp = []
    for i in indices:
        tmp.append(y_train[i])
    tmp = np.array(tmp)
    res, count = stats.mode(tmp, axis=1)
    counter = 0
    #return accuracy
    for i in range(len(res)):
        if res[i] == y_test[i]:
            counter = counter + 1
    return counter/n_test_sample

def compute_distances(X_train, X_test):
    dists = -2 * np.dot(X_test, X_train.T) + np.sum(X_train**2, axis=1) + np.sum(X_test**2, axis=1)[:, np.newaxis]
    return dists

# now lets test the model for linear models, that is, SVM and softmax
def linear_classifier_test(Wb, x_te, y_te, num_class,n_test):
    Wb = np.reshape(Wb, (-1, 1))
    dlen = len(x_te[0])
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * dlen)], (num_class, dlen))
    accuracy = 0;

    for i in range(n_test):
        # find the linear scores
        s = W @ x_te[i].reshape((-1, 1)) + b
        # find the maximum score index
        res = np.argmax(s)
        accuracy = accuracy + (res == y_te[i]).astype('uint8')

    return accuracy / n_test

# number of classes: this can be either 3 or 4
num_class = 4

# sigma controls the degree of data scattering. Larger sigma gives larger scatter
# default is 1.0. Accuracy becomes lower with larger sigma
sigma = 1.0

print('number of classes: ',num_class,' sigma for data scatter:',sigma)
if num_class == 4:
    n_train = 400
    n_test = 100
    feat_dim = 2
else:  # then 3
    n_train = 300
    n_test = 60
    feat_dim = 2

# generate train dataset
print('generating training data')
x_train, y_train = dg.generate(number=n_train, seed=None, plot=True, num_class=num_class, sigma=sigma)

# generate test dataset
print('generating test data')
x_test, y_test = dg.generate(number=n_test, seed=None, plot=False, num_class=num_class, sigma=sigma)

# set classifiers to 'svm' to test SVM classifier
# set classifiers to 'softmax' to test softmax classifier
# set classifiers to 'knn' to test kNN classifier
classifiers = 'softmax'

if classifiers == 'svm':
    print('training SVM classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(svm_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))
    print('testing SVM classifier...')

    Wb = result.x
    print('accuracy of SVM loss: ', linear_classifier_test(Wb, x_test, y_test, num_class,n_test)*100,'%')

elif classifiers == 'softmax':
    print('training softmax classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(cross_entropy_softmax_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))

    print('testing softmax classifier...')

    Wb = result.x
    print('accuracy of softmax loss: ', linear_classifier_test(Wb, x_test, y_test, num_class,n_test)*100,'%')

else:  # knn
    # k value for kNN classifier. k can be either 1 or 3.
    k = 3
    print('testing kNN classifier...')
    print('accuracy of kNN loss: ', knn_test(x_train, y_train, x_test, y_test, n_train, n_test, k)*100
          , '% for k value of ', k)
