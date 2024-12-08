import numpy as np
import pandas as pd 
from scipy.stats import multivariate_t

''' 
Same artificial data as in the conformal selection paper, 8 settings (NEW)
'''
def gen_data_Jin2023(setting, n, sig, dim=20):
    if setting == 1:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.25 + X[:,3]) + (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.25)
        Y = mu_x + np.random.normal(size=n) * sig
        return X, Y
    
    if setting == 2:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) - 1) * 2
        Y = mu_x + np.random.normal(size=n) * 1.5 * sig 
        return X, Y
    
    if setting == 3:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] > 0) * (X[:,3] > 0.5) * (0.25 + X[:,3]) + (X[:,0] * X[:,1] <= 0) * (X[:,3] < -0.5) * (X[:,3] - 0.25)
        Y = mu_x + np.random.normal(size=n) * (5.5 - abs(mu_x)) / 2 * sig
        return X, Y

    if setting == 4:
        X = np.random.uniform(low=-1, high=1, size=n*dim).reshape((n,dim))
        mu_x = (X[:,0] * X[:,1] + X[:,2] ** 2 + np.exp(X[:,3] - 1) - 1) * 2
        Y = mu_x + np.random.normal(size=n) * (5.5 - abs(mu_x)) / 2 * sig
        return X, Y
    
def gen_data_Liang2024(setting, n, sig, dim=300):
    # get a dim by dim identity matrix
    I_d = np.zeros((dim, dim))
    np.fill_diagonal(I_d, 1)

    if setting == 1:
        X = np.random.multivariate_normal([0] * dim, I_d, size=n)
        mask = np.array([0 if i % 20 != 0 else 1 for i in range(dim)])
        eps = np.random.normal(size=n)
        Y = eps * sig + X @ mask
        return X, Y
    
    if setting == 2:
        X = np.random.multivariate_normal([0] * dim, I_d, size=n)
        mask = np.array([0 if i % 20 != 0 else 1 for i in range(dim)])
        eps = np.random.standard_t(df=3, size=n)
        Y = eps * sig + X @ mask
        return X, Y
    
    if setting == 3:
        X = np.random.multivariate_normal([0] * dim, I_d, size=n)
        mask = np.array([1 / dim for i in range(dim)])
        eps = np.random.normal(size=n) / dim
        Y = eps * sig + X @ mask
        return X, Y
    
    if setting == 4:
        rv = multivariate_t([0] * dim, I_d, df=3)
        # generating multivariate t-distributed data is slower than calling np.random functions
        X = rv.rvs(size=n)
        mask = np.array([0 if i % 20 != 0 else 1 for i in range(dim)])
        eps = np.random.normal(size=n)
        Y = eps * sig + X @ mask
        return X, Y
    
def eval(Y, rejected, lower, higher):
    """
    Evaluate the selection performace: power and FDP. The region (lower, higher) corresponds to the null hypothesis.
    """
    true_reject = np.sum((lower < Y) & (Y < higher))
    if len(rejected) == 0:
        fdp = 0
        power = 0
    else:
        fdp = np.sum((lower >= Y[rejected]) | (Y[rejected] >= higher)) / len(rejected)
        power = np.sum((lower < Y[rejected]) & (Y[rejected] < higher)) / true_reject if true_reject != 0 else 0
    return fdp, power

def BH(pvals, q):
    """
    Given a list of p-values and nominal FDR level q, apply BH procedure to get a rejection set.
    """
    ntest = len(pvals)
         
    df_test = pd.DataFrame({"id": range(ntest), "pval": pvals}).sort_values(by='pval')
    
    df_test['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
    idx_smaller = [j for j in range(ntest) if df_test.iloc[j,1] <= df_test.iloc[j,2]]
    
    if len(idx_smaller) == 0:
        return np.array([])
    else:
        idx_sel = np.array(df_test.index[range(np.max(idx_smaller) + 1)])
        return idx_sel
    
def eBH(evals, q):
    """
    Given a list of e-values and nominal FDR level q, apply base eBH procedure (no pruning) to get a rejection set.
    """
    return BH(1 / evals, q)