# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 23:20:11 2018

@author: T.Yonezu
"""

import numpy as np
from dnn_func import sigmoid, dsigmoid, ReL, dReL, id_map, softmax, get_dif, output, sq_error, cross_entropy
import matplotlib.pyplot as plt
import seaborn as sns

# fix seed
#np.random.seed(0)

# Problem
_pbm = "reg" # reg, multi

# Training Data
N = 100
d = 4
X = np.random.normal(loc=0.0, scale=1.0, size=(d,N))

# Test Data
#X1 = np.random.normal(loc=0.0, scale=1.0, size=(d,N))

if _pbm == "reg":
    func = [np.nan, id_map, id_map]
    beta = np.array( [5]*2 + [0]*2 )[:,np.newaxis]
    D = np.dot(X.T,beta).T
    ef = sq_error
    
if _pbm == "multi":
    func = [np.nan, sigmoid, softmax]
    _qn = int(N/2.)
    _qd = int(d/2.)
    X[:_qd,:_qn] += np.ones((_qd,_qn))*3
    D = np.c_[ np.c_[np.ones(_qn), np.zeros(_qn)].T, 
                     np.c_[np.zeros(_qn), np.ones(_qn)].T]
    ef = cross_entropy

dim_in = len(X)
dim_out = len(D)

# Hyper Parameter
_ite = 10**2
b_size = 1
num_l = 2
num_u = [dim_in, 20, dim_out]
rate = 10**(-1)*5
lam = 10**(-4)

#####################
## mini_batch
#####################
Nt = int(N/b_size)
Dt = [ np.arange(i*Nt,(i+1)*Nt,1) for i in range(b_size) ]
W = [np.nan] + [ np.random.normal(size=(num_u[i+1], num_u[i])) for i in range(num_l)]
b = [np.nan] + [ np.zeros((num_u[i+1],1)) for i in range(num_l) ]
ones = np.ones(Nt)[:,np.newaxis]

# Learning
train_error = []
#class_error = []
for _s in range(_ite):
    for _t in range(b_size):
        U = [np.nan] + [0]*num_l
        Z = [ X[:, Dt[_t]] ] + [0]*num_l
        Dsub = D[:, _t*Nt:(_t+1)*Nt]
        Delta = [np.nan] + [0]*num_l
        
        for _l in range(1, num_l+1):
            U[_l] = np.dot(W[_l], Z[_l-1]) + np.dot(b[_l], ones.T)
            Z[_l] = func[_l]( U[_l] )
        
        Delta[num_l] = Z[num_l] - Dsub
        
        for _l in np.arange(num_l-1, 0, -1):
            tmp = get_dif(func[_l], U[_l])
            Delta[_l] = tmp * np.dot(W[_l+1].T, Delta[_l+1])
        
        for _l in np.arange(1, num_l+1):
            Delta_W = - rate * ( 1./Nt * np.dot(Delta[_l], Z[_l-1].T) + lam * W[_l] )
            W[_l] += Delta_W
            Delta_b = - rate * 1./Nt * np.dot(Delta[_l], ones)
            b[_l] += Delta_b
    
    y = output(X, W, b, func, num_l)
#    class_error.append( ( np.sum( (y>=0.5) == D) / 2.) / N )
    train_error.append( ef(D,y) )


plt.plot((train_error))
plt.show()
#plt.plot(class_error)
#plt.show()