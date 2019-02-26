# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:51:01 2018

@author: T.Yonezu
"""
import numpy as np

""" Activation Function """
# Sigmoid
def sigmoid(u):
    return 1./(1+np.exp(-u))

def dsigmoid(u):
    return sigmoid(u) * (1 - sigmoid(u))

# Softmax
def softmax(u):
    sub = u - np.max(u, axis=0)
    return np.exp( sub ) / np.sum(np.exp(sub), axis=0)

# Rectified Linear Unit
def ReL(u):
    return u*(u>0)

def dReL(u):
    return 1*(u>0)

# Identity Mapping
def id_map(u):
    return u

# Get Differential
def get_dif(func, u):
    if func == ReL:
        tmp = dReL( u )
    if func == id_map:
        tmp = np.ones( np.shape(u) )
    if func == sigmoid:
        tmp = dsigmoid( u )
    return tmp

""" Loss Function """
def sq_error(d, y):
    return np.sum( (y-d)**2 ) / 2.

def log_likelihood(d, y):
    return - np.sum( d * np.log(y) + (1 - d) * np.log(1 - y) )

def cross_entropy(d, y):
    return - np.sum( d * np.log(y) )

""" Others """
def size_check(X):
    for _s in X:
        print(np.shape(_s))
        
def output(X, W, b, func, num_l):
    _,N = np.shape(X)
    ones = np.ones(N)[:,np.newaxis]
    Z = np.copy(X)
    for l in range(1, num_l+1):
        U = np.dot(W[l], Z) + np.dot(b[l], ones.T)
        Z = func[l]( U )
    return Z

def make_mini_batch(train_data, Nt, T):
    inputs  = np.empty(0)
    outputs = np.empty(0)
    for _ in range(Nt):
        ind   = np.random.randint(0, len(train_data) - T)
        part    = train_data[ind : ind + T]
        inputs  = np.append(inputs, part[:, 0])
        outputs = np.append(outputs, part[-1, 1])
    inputs  = inputs.reshape(-1, T, 1)
    outputs = outputs.reshape(-1, 1)
    return (inputs, outputs)
