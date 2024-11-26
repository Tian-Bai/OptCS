import numpy as np
import pandas as pd 
from scipy.stats import multivariate_t
    
'''
Evaluate the selection performace: power and FDP. The region (lower, higher) corresponds to the null hypothesis
'''
def eval(Y, rejected, lower, higher):
    true_reject = np.sum((lower < Y) & (Y < higher))
    if len(rejected) == 0:
        fdp = 0
        power = 0
    else:
        fdp = np.sum((lower >= Y[rejected]) | (Y[rejected] >= higher)) / len(rejected)
        power = np.sum((lower < Y[rejected]) & (Y[rejected] < higher)) / true_reject if true_reject != 0 else 0
    return fdp, power

''' 
Given a list of p-values and nominal FDR level q, apply BH procedure to get a rejection set.
'''
def BH(pvals, q):
    ntest = len(pvals)
         
    df_test = pd.DataFrame({"id": range(ntest), "pval": pvals}).sort_values(by='pval')
    
    df_test['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
    idx_smaller = [j for j in range(ntest) if df_test.iloc[j,1] <= df_test.iloc[j,2]]
    
    if len(idx_smaller) == 0:
        return np.array([])
    else:
        idx_sel = np.array(df_test.index[range(np.max(idx_smaller) + 1)])
        return idx_sel
    
'''
Given a list of e-values and nominal FDR level q, apply base eBH procedure (no pruning) to get a rejection set.
'''
def eBH(evals, q):
    return BH(1 / evals, q)