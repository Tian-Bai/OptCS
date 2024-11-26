import numpy as np
import pandas as pd 
import random
import sys
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.svm import SVR
from utils import gen_data_Jin2023, eval, BH, eBH
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import argparse
import copy
from imblearn.over_sampling import RandomOverSampler

def LOO_train(mdl, Xtotal, Ytotal, LOO_range, Xtest, strategy, m=None):
    """
    Conduct leave-one-out training and return predicted results.
    
    Parameters: 
    -- mdl: model to be fitted
    -- Xtotal, Ytotal: the total training data. For each time, we fit the model using the total data minus one sample.
    -- LOO_range: the indices in (Xtotal, Ytotal) that is left out (one at a time)
    -- Xtest: Samples where we predict Y using the LOO-trained models.
    -- strategy: Strategy to use when fitting the model
    -- m: length of testing (unlabelled data)

    Returns:
    -- pred_test: (ntest, ntotal)
       pred_test[i][j] is the predicted results of X_i (in Xtest) using the j-th LOO model.
    -- pred_loo: (ntotal)
       pred_loo[i] is the predicted results of the left-out sample using the i-th LOO model.
    """
    n_total = len(LOO_range)
    pred_loo = np.zeros(n_total)
    if Xtest is not None:
        n_test = Xtest.shape[0]
        pred_test = np.zeros((n_test, n_total))
    else:
        pred_test = None

    for ii in range(n_total):
        Xtrain = np.delete(Xtotal, LOO_range[ii], axis=0)
        Ytrain = np.delete(Ytotal, LOO_range[ii])
        if strategy == 'oversample':
            ros = RandomOverSampler(random_state=42) # fix seed here
            Xtrain, Ytrain = ros.fit_resample(Xtrain, Ytrain)
        if strategy == 'remove_m':
            zero_idx = np.where(Ytrain == 0)[0]
            remove_idx = np.random.choice(zero_idx, size=m-1, replace=False)
            Xtrain, Ytrain = np.delete(Xtrain, remove_idx, axis=0), np.delete(Ytrain, remove_idx)
        if strategy == 'remove_half':
            zero_idx = np.where(Ytrain == 0)[0]
            remove_idx = np.random.choice(zero_idx, size=(m-1) // 2, replace=False)
            Xtrain, Ytrain = np.delete(Xtrain, remove_idx, axis=0), np.delete(Ytrain, remove_idx)
        if strategy == 'remove_prob':
            # first fit a binary classifier
            classifier = RandomForestClassifier()
            classifier.fit(Xtrain, Ytrain)
            p = classifier.predict_proba(Xtrain)[:, 0]  # predicted 0 probability
            r = np.where(Ytrain == 0, np.minimum(m / len(Xtotal) / p, 1), 0) # removal prob
            mask = np.random.rand(len(Xtrain)) >= r
            Xtrain, Ytrain = Xtrain[mask], Ytrain[mask]
        if strategy == 'remove_dist':
            idx = np.random.choice(len(Xtrain), size=m-1)
            X_sample, Y_sample = Xtrain[idx], Ytrain[idx]

            # first remove the zeros
            sampled_zero_idx = idx[np.where(Y_sample == 0)[0]]
            sampled_one_idx = idx[np.where(Y_sample == 1)[0]]
            rem_zero_idx =np.setdiff1d(np.where(Ytrain == 0)[0], set(sampled_zero_idx))
            
            if len(sampled_zero_idx) < m-1:
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(Xtrain[rem_zero_idx])
                _, nearest_idx = nbrs.kneighbors(Xtrain[sampled_one_idx])
                remove_idx = np.concatenate((sampled_zero_idx, rem_zero_idx[nearest_idx.flatten()]))
            else:
                remove_idx = sampled_zero_idx

            Xtrain, Ytrain = np.delete(Xtrain, remove_idx, axis=0), np.delete(Ytrain, remove_idx)

        mdl.fit(Xtrain, Ytrain)

        if Xtest is not None:
            pred_test[:, ii] = mdl.predict(Xtest) 
        pred_loo[ii] = mdl.predict(Xtotal[LOO_range[ii]].reshape((1, -1)))[0] # predict only the left out sample
        
    return pred_test, pred_loo

parser = argparse.ArgumentParser(description='Parser.')
parser.add_argument('nlabel', type=int)
parser.add_argument('ntest', type=int)
parser.add_argument('modelid', type=int)
parser.add_argument('setting', type=int)  
parser.add_argument('seed', type=int)  
parser.add_argument('sigma', type=float)
parser.add_argument('dim', type=int)

args = parser.parse_args()

modelid = args.modelid
setting = args.setting 
seed = args.seed
nlabel = args.nlabel
ntest = args.ntest
sig = args.sigma
dim = args.dim

q_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
split_ratio_list = [0.1, 0.25, 0.5, 0.75, 0.9]

all_res = pd.DataFrame()

random.seed(seed)
np.random.seed(seed)
 
Xlabel, Ylabel = gen_data_Jin2023(setting, nlabel, sig, dim)
Xtest, Ytest = gen_data_Jin2023(setting, ntest, sig, dim) 
    
if modelid == 1:
    init_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)
elif modelid == 2:
    init_model = RandomForestRegressor(max_depth=5, random_state=0)
else:
    init_model = SVR(kernel='rbf', gamma=0.1)

""" LOO version 1.1 and 1.2 - without / with oversampling + strategies to remove null points """

LOO_11_pvals, LOO_12_pvals, LOO_13_pvals, LOO_14_pvals, LOO_15_pvals, LOO_16_pvals = np.zeros(ntest), np.zeros(ntest), np.zeros(ntest), np.zeros(ntest), np.zeros(ntest), np.zeros(ntest)
Xtotal = np.concatenate([Xlabel, Xtest])
Ytotal = np.concatenate([1 * (Ylabel > 0), np.zeros(ntest)]) 

# LOO 1.1
_, pred_all = LOO_train(init_model, Xtotal, Ytotal, np.arange(len(Xtotal)), None, strategy='None')

pred_labeled = pred_all[range(nlabel)]
pred_test = pred_all[range(nlabel, nlabel+ntest)]

calib_scores = 1000 * (Ylabel > 0) - pred_labeled
test_scores = -pred_test
for j in range(ntest):
    LOO_11_pvals[j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)

# LOO 1.2
_, pred_all = LOO_train(init_model, Xtotal, Ytotal, np.arange(len(Xtotal)), None, strategy='oversample')

pred_labeled = pred_all[range(nlabel)]
pred_test = pred_all[range(nlabel, nlabel+ntest)]

calib_scores = 1000 * (Ylabel > 0) - pred_labeled
test_scores = -pred_test
for j in range(ntest):
    LOO_12_pvals[j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)

# LOO 1.3 - not very useful
_, pred_all = LOO_train(init_model, Xtotal, Ytotal, np.arange(len(Xtotal)), None, strategy='remove_m', m=ntest)

pred_labeled = pred_all[range(nlabel)]
pred_test = pred_all[range(nlabel, nlabel+ntest)]

calib_scores = 1000 * (Ylabel > 0) - pred_labeled
test_scores = -pred_test
for j in range(ntest):
    LOO_13_pvals[j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)

# LOO 1.4 - not very useful
_, pred_all = LOO_train(init_model, Xtotal, Ytotal, np.arange(len(Xtotal)), None, strategy='remove_half', m=ntest)

pred_labeled = pred_all[range(nlabel)]
pred_test = pred_all[range(nlabel, nlabel+ntest)]

calib_scores = 1000 * (Ylabel > 0) - pred_labeled
test_scores = -pred_test
for j in range(ntest):
    LOO_14_pvals[j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)

# LOO 1.5
_, pred_all = LOO_train(init_model, Xtotal, Ytotal, np.arange(len(Xtotal)), None, strategy='remove_prob', m=ntest)

pred_labeled = pred_all[range(nlabel)]
pred_test = pred_all[range(nlabel, nlabel+ntest)]

calib_scores = 1000 * (Ylabel > 0) - pred_labeled
test_scores = -pred_test
for j in range(ntest):
    LOO_15_pvals[j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)

# LOO 1.6
_, pred_all = LOO_train(init_model, Xtotal, Ytotal, np.arange(len(Xtotal)), None, strategy='remove_dist', m=ntest)

pred_labeled = pred_all[range(nlabel)]
pred_test = pred_all[range(nlabel, nlabel+ntest)]

calib_scores = 1000 * (Ylabel > 0) - pred_labeled
test_scores = -pred_test
for j in range(ntest):
    LOO_16_pvals[j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)

""" LOO version 2 - considering ntest subproblems, each having 1 unlabelled sample """

LOO_2_pvals = np.zeros(ntest)
aux_size = np.zeros((ntest, len(q_list)))
aux_avg_size = np.zeros((ntest, len(q_list)))

for j in range(ntest): 
    Xtotal = np.concatenate([Xlabel, Xtest[j,:].reshape(1, dim)], axis=0) 
    Ytotal = np.concatenate([1 * (Ylabel > 0), [0]]) 
    Xtest_j = np.delete(Xtest, j, 0)
    pred_test, pred_loo = LOO_train(init_model, Xtotal, Ytotal, np.arange(len(Xtotal)), Xtest_j, strategy='None') # an option is to use a smaller LOO_range
    
    # calib_score 
    calib_scores = 1000 * (Ytotal[:-1] > 0) - pred_loo[:-1]
    test_score_j = - pred_loo[-1]
    
    # compute p-value 
    LOO_2_pvals[j] = (np.sum(calib_scores < test_score_j) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_score_j) + 1)) / (len(calib_scores) + 1)
    aux_pvals = np.zeros(ntest)
    # aux_pvals_avg = np.zeros(ntest)
    loo_scores = np.concatenate([calib_scores, np.array([test_score_j])], axis=0)

    for ell in range(ntest - 1): 
        # aux_pvals_avg[ell] = (np.sum(loo_scores < -np.mean(pred_test[ell, :])) + np.random.uniform(size=1)[0] * (np.sum(loo_scores == -np.mean(pred_test[ell, :])))) / (len(calib_scores) + 1)
        aux_pvals[ell] = (np.sum(loo_scores < -pred_test[ell, -1]) + np.random.uniform(size=1)[0] * (np.sum(loo_scores == -pred_test[ell, -1]))) / (len(calib_scores) + 1)
    
    for qid, q in enumerate(q_list):
        aux_sel = BH(aux_pvals, q)
        aux_size[j, qid] = len(aux_sel)
        # aux_sel_avg = BH(aux_pvals_avg, q)
        # aux_avg_size[j, qid] = len(aux_sel_avg)

""" Baseline: split method with different split ratio """

split_pvals = np.zeros((len(split_ratio_list), ntest))

for s_idx, split_ratio in enumerate(split_ratio_list):
    split_reg = copy.deepcopy(init_model)

    # split
    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xlabel, Ylabel, train_size=split_ratio, random_state=42)

    split_reg.fit(Xtrain, 1 * (Ytrain > 0))
    split_calib_scores = 1000 * (Ycalib > 0) - split_reg.predict(Xcalib)
    split_test_scores = -split_reg.predict(Xtest)

    for j in range(ntest):
        split_pvals[s_idx][j] = (np.sum(split_calib_scores < split_test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(split_calib_scores == split_test_scores[j]) + 1)) / (len(split_calib_scores) + 1)

""" Oracle: what if we have access to all labelled data both for training and calibration? """

orc_Xtrain, orc_Ytrain = gen_data_Jin2023(setting, nlabel, sig, dim) 
orc_Xcalib, orc_Ycalib = gen_data_Jin2023(setting, nlabel, sig, dim) 

orc_reg = copy.deepcopy(init_model)
orc_reg.fit(orc_Xtrain, 1 * (orc_Ytrain > 0))
orc_calib_scores = 1000 * (orc_Ycalib > 0) - orc_reg.predict(orc_Xcalib)
orc_test_scores = - orc_reg.predict(Xtest)

orc_pvals = np.zeros(ntest)
for j in range(ntest):
    orc_pvals[j] = (np.sum(orc_calib_scores < orc_test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(orc_calib_scores == orc_test_scores[j]) + 1)) / (len(orc_calib_scores) + 1)

""" conduct selection for all methods """

beta = 0.9 # a coefficient for pruning

for qid, q in enumerate(q_list): 
    LOO_11_sel = BH(LOO_11_pvals, q)
    LOO_12_sel = BH(LOO_12_pvals, q)
    LOO_13_sel = BH(LOO_13_pvals, q)
    LOO_14_sel = BH(LOO_14_pvals, q)
    LOO_15_sel = BH(LOO_15_pvals, q)
    LOO_16_sel = BH(LOO_16_pvals, q)
    LOO_2_sel = BH(LOO_2_pvals, q)

    # LOO_2 with e-values
    evalues = 1 * (LOO_2_pvals <= (q * aux_size[:, qid] / ntest)) / (q * aux_size[:, qid] / ntest) + 1e-8
    LOO_2e_dtm_sel = eBH(evalues, q)
    LOO_2e_homo_sel = eBH(evalues / np.random.uniform(size=1), q)
    LOO_2e_hete_sel = eBH(evalues / np.random.uniform(size=ntest), q)

    # LOO_2 with e-values computed with beta=0.9
    evalues_9 = 1 * (LOO_2_pvals <= q * aux_size[:, qid] * beta / ntest) / (q * aux_size[:, qid] * beta / ntest) + 1e-8
    LOO_2e9_dtm_sel = eBH(evalues, q)
    LOO_2e9_homo_sel = eBH(evalues / np.random.uniform(size=1), q)
    LOO_2e9_hete_sel = eBH(evalues / np.random.uniform(size=ntest), q)
    
    # split
    split_sel = []
    for s_idx, split_ratio in enumerate(split_ratio_list):
        split_sel.append(BH(split_pvals[s_idx], q))

    # oracle
    oracle_sel = BH(orc_pvals, q)

    # evaluate them
    LOO_11_FDP, LOO_11_power = eval(Ytest, LOO_11_sel, 0, np.inf)
    LOO_12_FDP, LOO_12_power = eval(Ytest, LOO_12_sel, 0, np.inf)
    LOO_13_FDP, LOO_13_power = eval(Ytest, LOO_13_sel, 0, np.inf)
    LOO_14_FDP, LOO_14_power = eval(Ytest, LOO_14_sel, 0, np.inf)
    LOO_15_FDP, LOO_15_power = eval(Ytest, LOO_15_sel, 0, np.inf)
    LOO_16_FDP, LOO_16_power = eval(Ytest, LOO_16_sel, 0, np.inf)
    LOO_2_FDP, LOO_2_power = eval(Ytest, LOO_2_sel, 0, np.inf)

    LOO_2e_dtm_FDP, LOO_2e_dtm_power = eval(Ytest, LOO_2e_dtm_sel, 0, np.inf)
    LOO_2e_homo_FDP, LOO_2e_homo_power = eval(Ytest, LOO_2e_homo_sel, 0, np.inf)
    LOO_2e_hete_FDP, LOO_2e_hete_power = eval(Ytest, LOO_2e_hete_sel, 0, np.inf)

    LOO_2e9_dtm_FDP, LOO_2e9_dtm_power = eval(Ytest, LOO_2e9_dtm_sel, 0, np.inf)
    LOO_2e9_homo_FDP, LOO_2e9_homo_power = eval(Ytest, LOO_2e9_homo_sel, 0, np.inf)
    LOO_2e9_hete_FDP, LOO_2e9_hete_power = eval(Ytest, LOO_2e9_hete_sel, 0, np.inf)

    oracle_FDP, oracle_power = eval(Ytest, oracle_sel, 0, np.inf)
    
    df_res = pd.DataFrame({
        'LOO_11_FDP': [LOO_11_FDP],
        'LOO_11_power': [LOO_11_power],
        'LOO_12_FDP': [LOO_12_FDP],
        'LOO_12_power': [LOO_12_power],
        'LOO_13_FDP': [LOO_13_FDP],
        'LOO_13_power': [LOO_13_power],
        'LOO_14_FDP': [LOO_14_FDP],
        'LOO_14_power': [LOO_14_power],
        'LOO_15_FDP': [LOO_15_FDP],
        'LOO_15_power': [LOO_15_power],
        'LOO_16_FDP': [LOO_16_FDP],
        'LOO_16_power': [LOO_16_power],
        'LOO_2_FDP': [LOO_2_FDP],
        'LOO_2_power': [LOO_2_power],
        'LOO_2e_dtm_FDP': [LOO_2e_dtm_FDP],
        'LOO_2e_dtm_power': [LOO_2e_dtm_power],
        'LOO_2e_homo_FDP': [LOO_2e_homo_FDP],
        'LOO_2e_homo_power': [LOO_2e_homo_power],
        'LOO_2e_hete_FDP': [LOO_2e_hete_FDP],
        'LOO_2e_hete_power': [LOO_2e_hete_power],
        'LOO_2e9_dtm_FDP': [LOO_2e9_dtm_FDP],
        'LOO_2e9_dtm_power': [LOO_2e9_dtm_power],
        'LOO_2e9_homo_FDP': [LOO_2e9_homo_FDP],
        'LOO_2e9_homo_power': [LOO_2e9_homo_power],
        'LOO_2e9_hete_FDP': [LOO_2e9_hete_FDP],
        'LOO_2e9_hete_power': [LOO_2e9_hete_power],
        'Oracle_FDP': [oracle_FDP],
        'Oracle_power': [oracle_power],
        'q': [q],
        'setting': [setting],
        'model': [['XGB', 'RF', 'SVR'][modelid - 1]], 
        'dim': [dim],
        'seed': [seed],
        'sigma': [sig],
        'ntest': [ntest],
        'nlabel': [nlabel]
    })

    for s_idx, split_ratio in enumerate(split_ratio_list):
        split_FDP, split_power = eval(Ytest, split_sel[s_idx], 0, np.inf)
        df_res[f'Split_{split_ratio:.2f}_FDP'] = split_FDP
        df_res[f'Split_{split_ratio:.2f}_power'] = split_power

    all_res = pd.concat((all_res, df_res))

if not os.path.exists('results'):
    os.makedirs('results')

all_res.to_csv(os.path.join('results', f"LOO_Jin, setting={setting}, dim={dim}, sigma={sig}, modelid={modelid}, seed={seed}, ntest={ntest}, nlabel={nlabel}.csv"))
