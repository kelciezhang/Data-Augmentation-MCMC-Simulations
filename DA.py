import numpy as np
from polyagamma import random_polyagamma
from numpy.linalg import inv
import sys
import os
import pandas as pd
from scipy.optimize import bisect
from scipy.stats import truncnorm, norm
import scipy

path=os.getcwd()

# Parameters
link=sys.argv[1] # link function: "logit" or "probit"
setting=sys.argv[2] # scenario info: "joint", "ngrow", or "dgrow", correspoding to scenario 1, 2, or 3 in the paper respectively
target_ib=int(sys.argv[3]) # imbalance factor * 100
lp=int(sys.argv[4]) # # replicate number, recall we want to perform the sampling 100 times and get average autocorrelation time



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
seed = lp*100 # random seed 

# Utils
def stable_multivariate_normal(mu, Sigma):
    u_,d_,ut_ = scipy.linalg.svd(Sigma,lapack_driver='gesvd')
    sigma_sqrt=np.matmul(u_,np.diag(np.sqrt(d_)))
    return np.matmul(sigma_sqrt,np.random.randn(d))+mu

def find_a(X, target_ib, left=-10, right=150):
    global link
    n = X.shape[0]
    def test(a):
        np.random.seed(seed)
        theta_m = np.concatenate((np.array([a]),np.zeros(d-1))) #mean of theta truth
        theta_v = np.diag(np.concatenate(([0],np.ones(d-1)))) # variance of theta truth 
        theta = np.random.multivariate_normal(theta_m, theta_v)
        if link=='probit':
            p = norm.cdf(np.matmul(X,theta))
        elif link=='logit':
            p = 1/(1+np.exp(-np.matmul(X,theta)))
        else:
            raise('Error!')
        Y = np.random.binomial(n=1,p=p)
        return np.sum(Y)/n - target_ib/100
    if target_ib!=100:
        return bisect(test, left, right)
    else: 
        for a in range(150):
            if test(a)==0:
                return a  
            
def kernel_DAlogit(theta):
    global B, b, d, n, X, Y
    omega = random_polyagamma(z=X.dot(theta), random_state=seed)
    Sigma = inv(np.matmul(np.matmul(X.T,np.diag(omega)),X) + inv(B))
    mu = np.matmul(Sigma, np.matmul(X.T,Y-0.5)+np.matmul(inv(B),b))
    return stable_multivariate_normal(mu, Sigma)

def kernel_DAprobit(theta):
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
    b = np.zeros(d) # prior mean
    B = np.eye(d) # prior variance
    output_path = path + link + '/ib_' + str(target_ib) + '/n_' + str(n) + '/d_' + str(d) + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data_file = 'data_'+link+'_ib' + str(target_ib) + '_n' + str(n) + '_d' + str(d) + '.npy'
    result_file = 'tauto_'+link+'_ib' + str(target_ib) + '_n' + str(n) + '_d' + str(d)+'_'+str(lp)+'.npy'
    if os.path.exists(output_path + result_file):
        continue

    np.random.seed(seed)
    X = np.concatenate((np.ones(n).reshape(n,1),np.random.multivariate_normal(np.zeros(d-1),np.eye(d-1),n)),axis=1)
         
    a = find_a(X,target_ib) #imbalance
    # Generate data and coefficients
    np.random.seed(seed)
    theta_m = np.concatenate((np.array([a]),np.zeros(d-1))) #mean of theta truth
    theta_v = np.diag(np.concatenate(([0],np.ones(d-1)))) # variance of theta truth 
    theta = np.random.multivariate_normal(theta_m, theta_v)
    if link=='probit':
        p = norm.cdf(np.matmul(X,theta))
    elif link=='logit':
        p = 1/(1+np.exp(-np.matmul(X,theta)))
    else:
        raise('Error!')
    Y = np.random.binomial(n=1,p=p)
    B_eigen = np.linalg.eig(inv(B))[0]
    samplecov_eigen = np.linalg.eig(np.matmul(X.T, X)/n)[0]
    L = np.max(B_eigen) + 0.25*n*np.max(samplecov_eigen)
    m = np.min(B_eigen)
    kappa = L/m


    np.random.seed(seed)   
    trace = np.zeros((maxit+1, d))
    theta0 = np.concatenate((np.array([a]), np.zeros(d-1)))
    trace[0,:] = theta0
    for i in range(1, maxit+1):
        if link=='probit':
            theta0 = kernel_DAprobit(theta0)
        elif link=='logit':
            theta0 = kernel_DAlogit(theta0)
        else:
            raise('Error!')
        trace[i,:] = theta0


    burnin = 25000
    corr_list = []
    for lag in range(1,20000):
        corr_list.append(np.corrcoef(trace[burnin+lag:,1], trace[burnin:-lag,1])[1,0])
    t = pd.Series(corr_list)[pd.Series(corr_list)<=0].index[0]
    t_auto=1+2*np.sum(pd.Series(corr_list).iloc[0:t-1])



    result= {'n': n,\
             'd': d,\
             'a': a,\
             'lp':lp,\
             'ib': np.sum(Y)/n,\
             't_auto': t_auto
    }

    np.save(output_path+result_file, result)

