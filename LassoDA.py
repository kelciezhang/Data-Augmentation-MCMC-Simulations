"""
File name: LassoDA.py
Discription: The code are for running one replicate of Scenario 1, 2, or 3 in the paper using LassoDA. Command-line arguments are needed.
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
lp=int(sys.argv[2]) # replicate number 1-100
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

maxit = 50000 # maximum iteration
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
    
    # Make output directory
    output_path = path + 'lasso_result' + '/n_' + str(n) + '/d_' + str(d) + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Data generation
    np.random.seed(seed)
    X = random.multivariate_normal(zeros(d),eye(d),n)
    v_true = 1
    beta_true = random.multivariate_normal(zeros(d), diag(ones(d)))
    Y = matmul(X,beta_true) + random.multivariate_normal(zeros(n),eye(n)*v_true)
    Y = Y - mean(Y)

    xi = 1 # prior parameter for v
    alpha = 2 # prior parameter for v

    # Sampling
    trace = np.zeros((maxit+1, d+1))
    beta0 = np.zeros(d)
    v0 = v_true
    trace[0, 0] = v0
    trace[0,1:] = beta0
    for i in range(1, maxit+1):
        beta0, v0 = LassoDA_kernel(beta0, v0)
        trace[i,0] = v0
        trace[i,1:] = beta0
    
    # Calculate autocorrelation time
    burnin = 25000
    corr_list = []
    for lag in range(1,20000):
        corr_list.append(np.corrcoef(trace[burnin+lag:,1], trace[burnin:-lag,1])[1,0])
    t = pd.Series(corr_list)[pd.Series(corr_list)<=0].index[0]
    t_auto=1+2*np.sum(pd.Series(corr_list).iloc[0:t-1])

    burnin = 25000
    corr_list = []
    for lag in range(1,20000):
        corr_list.append(np.corrcoef(trace[burnin+lag:,0], trace[burnin:-lag,0])[1,0])
    t = pd.Series(corr_list)[pd.Series(corr_list)<=0].index[0]
    t_auto_v=1+2*np.sum(pd.Series(corr_list).iloc[0:t-1])

    # Save the result
    result= {'n': n,\
             'd': d,\
             'a': alpha,\
             'xi':xi,\
             'lam': lambda_,\
             'lp': lp, \
             't_auto': t_auto, \
             't_auto_v': t_auto_v
    }
    np.save(output_path+'tauto_' + 'n' + str(n) + '_d' + str(d) + '_' + str(lp) + '.npy', result)


