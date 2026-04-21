"""
File name: LassoDA.py
Discription: This code supports the simulations in the paper "Fast Mixing of Data Augmentation Algorithms: Bayesian Probit, Logit, and Lasso Regression" by Holden Lee and Kexin Zhang. This script runs a single replicate of Scenario 1, 2, or 3 from the paper using either LassoDA. Command-line arguments are required to specify the scenario and settings.
"""
import pandas as pd
import numpy as np
from numpy import *
from numpy.linalg import *
import os
import sys
from scipy.stats import invgauss, invgamma
import scipy

path=os.getcwd() + '/'

# Parameters
setting=sys.argv[1] # scenario info: "joint", "ngrow", or "dgrow", correspoding to scenario 1, 2, or 3 in the paper respectively
lp=int(sys.argv[2]) # replicate number: 1-100
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
burnin_len = 200 # burn-in time
lag_len = 100 # maximum lag
save_interval = 500 # interval of iterations at which intermediate values are saved

lambda_ = 1 # lambda the tuning paramter
seed = lp*100 # random seed

# Utils
def robust_invguass(b,v):
    """
    Generate inverse gaussian variables
    """
    global lambda_
    if b==0:
        return invgamma.rvs(a=0.5, scale=lambda_/2)
    return invgauss.rvs(mu=(lambda_*sqrt(v0)/abs(b))/lambda_**2, scale=lambda_**2)

def stable_multivariate_normal(mu, Sigma):
    """
    Generate multivariate normal variables
    """
    u_,d_,ut_ = scipy.linalg.svd(Sigma,lapack_driver='gesvd')
    sigma_sqrt=np.matmul(u_,np.diag(np.sqrt(d_)))
    return np.matmul(sigma_sqrt,np.random.randn(d))+mu


def LassoDA_kernel(beta0,v0):
    """
    One iteration of LassoDA
    """
    z=1/pd.Series(beta0).map(lambda b:robust_invguass(b,v0)).values
    v0 = invgamma.rvs(a=(n+2*alpha-1)/2, 
                   scale=xi+matmul(matmul(Y.T,eye(n)-matmul(matmul(X,inv(matmul(X.T,X)+diag(1/z))),X.T)),Y)/2)
    beta0 = stable_multivariate_normal(matmul(matmul(inv(matmul(X.T,X)+diag(1/z)), X.T),Y),
                              v0*inv(matmul(X.T,X)+diag(1/z)))
    return beta0, v0



# Main program
for nd_i in range(20):
    n = n_list[nd_i]
    d = d_list[nd_i]
    
    # Make the output directory
    output_path = path + 'lasso_result' + '/n_' + str(n) + '/d_' + str(d) + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    result_file = 'tauto_' + 'n' + str(n) + '_d' + str(d) + '_' + str(lp) + '.npy'
    if os.path.exists(output_path + result_file):
        continue
  
    # Data generation
    np.random.seed(seed)
    X = random.multivariate_normal(zeros(d),eye(d),n) / np.sqrt(d)
    X = X  - np.mean(X,axis=0)
    v_true = 1
    beta_true = random.multivariate_normal(zeros(d), diag(ones(d)))
    Y = matmul(X,beta_true) + random.multivariate_normal(zeros(n),eye(n)*v_true)
    Y = Y - mean(Y)

    xi = 1 # prior parameter for v
    alpha = 2 # prior parameter for v

    # Set initial values
    trace = np.zeros((maxit+1, d+1))
    beta0 = np.zeros(d)
    v0 = v_true
    trace[0, 0] = v0
    trace[0,1:] = beta0
    i_min = 0
    
    # Load intermediate values if any 
    temp_file = 'tauto_' + 'n' + str(n) + '_d' + str(d) + '_' + str(lp) + '_temp.npy'
    if os.path.exists(output_path+temp_file):
        temp_result = np.load(output_path+temp_file, allow_pickle=True).item()
        trace = temp_result['trace']
        i_min = temp_result['iter']
        v0 = trace[i_min,0]
        beta0 = trace[i_min,1:]
        X = temp_result['X']
        Y = temp_result['Y']
    
    # MCMC
    for i in range(1+i_min, maxit+1):
        beta0, v0 = LassoDA_kernel(beta0, v0)
        trace[i,0] = v0
        trace[i,1:] = beta0
        if i%save_interval == 0:
            temp_result= {'n': n,\
                 'd': d,\
                 'lp': lp, \
                 'trace': trace, \
                 'iter': i, \
                 'X': X, \
                 'Y': Y
            }
            np.save(output_path+temp_file, temp_result)
    os.remove(output_path+temp_file) 
            
    # Calculate autocorrelation time
    corr_mat=np.zeros((d,lag_len-1))
    burnin = burnin_len
    t_auto_list = []
    for d_index in range(d):
        corr_list = []
        for lag in range(1,lag_len):
            corr_list.append(np.corrcoef(trace[burnin+lag:, d_index+1], trace[burnin:-lag, d_index+1])[1,0])
        corr_mat[d_index,:]=corr_list
        if sum(pd.Series(corr_list)<=0)==0:
            t = lag_len-1
        else: 
            t = pd.Series(corr_list)[pd.Series(corr_list)<=0].index[0]
        t_auto=1+np.sum(pd.Series(corr_list).iloc[0:t])
        t_auto_list.append(t_auto)

    burnin = burnin_len
    corr_list = []
    for lag in range(1,lag_len):
        corr_list.append(np.corrcoef(trace[burnin+lag:,0], trace[burnin:-lag,0])[1,0])
    if sum(pd.Series(corr_list)<=0)==0:
        t = lag_len-1
    else: 
        t = pd.Series(corr_list)[pd.Series(corr_list)<=0].index[0]
    t_auto_v=1+np.sum(pd.Series(corr_list).iloc[0:t])

    # Save the result
    result= {'n': n,\
             'd': d,\
             'a': alpha,\
             'xi':xi,\
             'lam': lambda_,\
             'lp': lp, \
             'tmax': np.max(t_auto_list), \
             't_auto_list': t_auto_list, \
             't_auto_v': t_auto_v, \
             'corr_mat': corr_mat, \
             'corr_v': corr_list
    }
    np.save(output_path + result_file, result)


