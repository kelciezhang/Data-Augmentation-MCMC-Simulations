"""
File name: DA.py
Discription: This code supports the simulations in the paper "Fast Mixing of Data Augmentation Algorithms: Bayesian Probit, Logit, and Lasso Regression" by Holden Lee and Kexin Zhang. This script runs a single replicate of Scenario 1, 2, or 3 from the paper using either ProbitDA or LogitDA. Command-line arguments are required to specify the scenario and settings.
"""

import numpy as np
from polyagamma import random_polyagamma
from numpy.linalg import inv
import sys
import os
import pandas as pd
from scipy.optimize import bisect
from scipy.stats import truncnorm, norm
import scipy

path=os.getcwd() + '/'

# Parameters
link=sys.argv[1] # link function: "logit" or "probit"
setting=sys.argv[2] # scenario info: "joint", "ngrow", or "dgrow", correspoding to scenario 1, 2, or 3 in the paper respectively
lp=int(sys.argv[3]) # # replicate number: 1-100 

if setting=='joint':
    n_list = list(range(50,1001,50))
    d_list = list(range(50,1001,50))
elif setting=='ngrow':
    n_list = list(range(50,1001,50))
    d_list = [500]*20
elif setting=='dgrow':
    n_list = [500]*20
    d_list = list(range(50,1001,50))
else:
    raise "Unknown Setting"
    
maxit = 1000 # maximum iteration
#save_interval = 500 # interval of iterations at which intermediate values are saved
burnin_len = 200 # burn-in time
lag_len = 100 # maximum lag
seed = lp*100 # random seed 
imbalanced = True # Set this to be True for the worst-case simulations

# Utils
def stable_multivariate_normal(mu, Sigma):
    """
    Generate multivariate normal variables given mean and covariance matrix
    """
    u_,d_,ut_ = scipy.linalg.svd(Sigma,lapack_driver='gesvd')
    sigma_sqrt=np.matmul(u_,np.diag(np.sqrt(d_)))
    return np.matmul(sigma_sqrt,np.random.randn(d))+mu
            
def kernel_DAlogit(theta):
    """
    One iteration of LogitDA 
    """
    global B, b, d, n, X, Y
    omega = random_polyagamma(z=X.dot(theta), random_state=seed)
    Sigma = inv(np.matmul(np.matmul(X.T,np.diag(omega)),X) + inv(B))
    mu = np.matmul(Sigma, np.matmul(X.T,Y-0.5)+np.matmul(inv(B),b))
    return stable_multivariate_normal(mu, Sigma)

def kernel_DAprobit(theta):
    """
    One iteration of ProbitDA 
    """
    global B, b, d, n, X, Y
    h = np.matmul(X,theta)
    z = np.zeros(n)
    z[Y==1]=truncnorm.rvs(-h[Y==1], np.inf, loc=h[Y==1], scale=1,size=sum(Y==1))
    z[Y==0]=truncnorm.rvs(-np.inf, -h[Y==0], loc=h[Y==0], scale=1,size=sum(Y==0))
    Sigma = inv(np.matmul(X.T,X) + inv(B))
    mu = np.matmul(Sigma, np.matmul(X.T,z)+np.matmul(inv(B),b))
    return stable_multivariate_normal(mu, Sigma)

# Main program
for nd_i in range(20):
    n = n_list[nd_i]
    d = d_list[nd_i]
    
    # Make the output directory, skip if there is already result in the target directory
    output_path = path + link + '/n_' + str(n) + '/d_' + str(d) + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    result_file = 'tauto_' + link + '_n' + str(n) + '_d' + str(d)+'_'+str(lp)+'.npy'
    if os.path.exists(output_path + result_file):
        continue
        
    # Generate data
    b = np.zeros(d) # prior mean
    B = np.eye(d) # prior variance

    np.random.seed(seed)
    X = np.concatenate((np.ones(n).reshape(n,1),np.random.multivariate_normal(np.zeros(d-1),np.eye(d-1),n)/np.sqrt(d)),axis=1) 
    
    theta_m = np.concatenate((np.array([1]),np.zeros(d-1))) # mean of theta truth
    theta_v = np.diag(np.ones(d)) # variance of theta truth 
    theta = np.random.multivariate_normal(theta_m, theta_v)
    if link=='probit':
        p = norm.cdf(np.matmul(X,theta))
    elif link=='logit':
        p = 1/(1+np.exp(-np.matmul(X,theta)))
    else:
        raise('Error!')
    if imbalanced:
        Y = np.random.binomial(n=1,p=p)
    else:
        Y = np.random.binomial(n=1,p=np.ones(np.shape(p)))

    # Set initial values
    trace = np.zeros((maxit+1, d))
    theta0 = np.concatenate((np.array([1]), np.zeros(d-1)))
    trace[0,:] = theta0
    i_min = 0
    
    # MCMC
    for i in range(i_min+1, maxit+1):
        if link=='probit':
            theta0 = kernel_DAprobit(theta0)
        elif link=='logit':
            theta0 = kernel_DAlogit(theta0)
        else:
            raise('Error!')
        trace[i,:] = theta0
    
    # Calculate autocorrelation time
    corr_mat=np.zeros((d,lag_len-1))
    burnin = burnin_len 
    t_auto_list = []
    for d_index in range(d):
        corr_list = []
        for lag in range(1,lag_len):
            corr_list.append(np.corrcoef(trace[burnin+lag:,d_index], trace[burnin:-lag,d_index])[1,0])
        corr_mat[d_index,:]=corr_list
        if sum(pd.Series(corr_list)<=0)==0:
            t = lag_len-1
        else: 
            t = pd.Series(corr_list)[pd.Series(corr_list)<=0].index[0]
        t_auto=1+np.sum(pd.Series(corr_list).iloc[0:t])
        t_auto_list.append(t_auto)

    # Save the result
    result= {'n': n,\
             'd': d,\
             'lp':lp,\
             'ib':np.sum(Y)/n,\
             'tmax': np.max(t_auto_list), \
             't_auto_list': t_auto_list,\
             'corr_mat': corr_mat
    } 
    np.save(output_path+result_file, result)

