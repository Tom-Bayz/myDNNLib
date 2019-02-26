# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:30:09 2018

@author: T.Yonezu
"""

import numpy as np
from dnn_func import sigmoid, dsigmoid, ReL, dReL, id_map, softmax, get_dif, output, sq_error, make_mini_batch
import matplotlib.pyplot as plt
import seaborn as sns

# fix seed
np.random.seed(0)

# Problem
_pbm = "reg" # reg, multi

# Setting
_epochs = 10**2
T = 50
b_size = 100
dim_in = 1
dim_out = 1

# make data
data = np.load("data_sin.npy")
[X, D] = make_mini_batch(data, b_size, T)

# Hyper Parameter
num_l = 3
num_hu = 2 # number of hidden unit
rate = 10**(-2)
lam = 0.01

if _pbm == "reg":
    func1 = id_map
    func2 = id_map
    ef = sq_error

W_in = np.random.normal(scale=0.3, size=(num_hu, dim_in+1))
W_h = np.random.normal(scale=0.3, size=(num_hu, num_hu) )
W_out = np.random.normal(scale=0.3, size=(dim_out, num_hu+1) )
Ww_out = W_out[:, 1:]
W0_out = W_out[:, 0]

# Learning
train_error = []
for _e in range(_epochs):
    ind = np.random.choice(np.arange(b_size), 1)[0]
    Xsub = np.c_[np.ones(T), X[ind, :]]
    Dsub = np.r_[ X[ind, 1:], D[ind].reshape(-1,1) ]
    
    """ Forward """
    y = np.zeros((T,1))
    u = np.zeros((num_hu,T))
    z = np.zeros((num_hu,T+1))
    for _t in range(T):
        xt = Xsub[_t].reshape(-1, 1)
        zt = z[:, _t].reshape(-1, 1)
        # Cal u^t
        _tmp1 = ( np.dot(W_in, xt) + np.dot(W_h, zt) ).reshape(-1,)
        u[:, _t] = _tmp1
        # Cal z^t
        _tmp2 = func1( _tmp1 )
        z[:, _t+1] = _tmp2
        # Cal y^t
        y[_t] = func2( np.dot(Ww_out, _tmp2.reshape(-1, 1) ) + W0_out)
    
    """ Back """
    # Delta_out
    Delta_out = y - Dsub    
    # Delta^t
    Delta = np.zeros((T+1, num_hu))
    dif = get_dif(func1, u)
    for _t in np.arange(T-1, -1, -1):
        for _j in np.arange(num_hu):
            Delta[_t, _j] = np.dot(Ww_out[:, _j], Delta_out[_t]) + np.dot(W_h[_j], Delta[_t+1,:]) * dif[_j, _t]
    Delta = Delta[:T]
    
    Delta_W0_out = np.sum(Delta_out, axis=0)
    Delta_W_in = np.zeros((num_hu, dim_in+1))
    Delta_W_h = np.zeros((num_hu, num_hu))
    Delta_Ww_out = np.zeros((dim_out, num_hu))
    for _j in range(num_hu):
        for _i in range(dim_in+1):
            Delta_W_in[_j, _i] = np.sum( Delta[:, _j] * Xsub[:, _i] )
        
        for _h in range(num_hu):
            Delta_W_h[_j, _h] = np.sum( Delta[:, _j] * z[_h, :T] )        
        
        for _k in range(dim_out):
            Delta_Ww_out[_k, _j] = np.sum( Delta_out[:, _k] * z[_j, 1:] )
    
    """ Parameter Update """
    W_in += - rate * ( Delta_W_in + lam * W_in )
    W_h += - rate * ( Delta_W_h + lam * W_h )
    Ww_out += - rate * ( Delta_Ww_out + lam * Ww_out )
    W0_out += - rate * ( Delta_W0_out + lam * W0_out )
    
    train_error.append( ef(Dsub,y) )

plt.plot(train_error)
plt.show()