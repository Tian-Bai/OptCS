import numpy as np
import pandas as pd 
import random
import sys
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, QuantileRegressor
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from quantile_forest import RandomForestQuantileRegressor
from sklearn.svm import SVR
from utils import gen_data_Liang2024, eval, BH, eBH
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import argparse
import copy
from imblearn.over_sampling import RandomOverSampler

def Split_train_modsel(mdl_idx, Xtotal, Ytotal, split_ratio, Xtest):
    """
    Conduct regular split training and return predicted results.
    
    Parameters: 
    -- mdl_idx: model id to be fitted
    -- Xtotal, Ytotal: the total training data. For each time, we fit the model using the total data minus one sample.
    -- split_ratio: split_ratio for the Xtotal, Ytotal data
    -- Xtest: Samples where we predict Y using the LOO-trained models.
    
    Returns:
    -- calib_scores: calibration scores
    -- test_scores: test scores
    """
    # get the model
    if mdl_idx < 3: 
        qt = 0.25 * (mdl_idx + 1)
        mdl = QuantileRegressor(quantile=qt)
        features = np.random.choice(dim, dim // 10)
    elif mdl_idx in [3, 4]:
        mdl = LinearRegression()
        features = np.random.choice(dim, dim // 10)
    elif mdl_idx in [5, 6]:
        mdl = LinearRegression()
        features = np.random.choice(dim, dim // 5)

    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtotal, Ytotal, train_size=split_ratio)
    mdl.fit(Xtrain[:, features], Ytrain)

    if mdl_idx < 3:
        denom = 1
    else:
        Xsigma, Ysigma = gen_data_Liang2024(setting, 100, sig, dim)
        if mdl_idx in [3, 4, 5]:
            # decrease n_estimators for runtime
            denom = LinearRegression()
        if mdl_idx in [6, 7, 8]:
            denom = SVR(kernel='rbf', gamma=0.1)
        denom.fit(Xsigma, np.absolute(Ysigma - mdl.predict(Xsigma[:, features])))

    pred_test = mdl.predict(Xtest[:, features]) / (1 if isinstance(denom, int) else denom.predict(Xtest))
    pred_calib = mdl.predict(Xcalib[:, features]) / (1 if isinstance(denom, int) else denom.predict(Xcalib))

    calib_scores = 1000 * (Ycalib > 0) - pred_calib
    test_scores = -pred_test
    
    return calib_scores, test_scores

def Modsel_train_modsel(Xtotal, Ytotal, split_ratio1, split_ratio2, Xtest, q_list):
    """
    Conduct a greedy split-based model selection, then do regular split training and return predicted results.
    
    Parameters: 
    -- mdl_idx: model id to be fitted
    -- Xtotal, Ytotal: the total training data. For each time, we fit the model using the total data minus one sample.
    -- split_ratio1: % of data for model selection
    -- split_ratio2: after spliting the data for model selection, % of the remaining data for model training
    -- Xtest: Samples where we predict Y using the LOO-trained models.
    -- q_list: nominal levels
    
    Returns:
    -- selections: (len(q_list)), selection sets
    """
    Xmodsel, Xreg, Ymodsel, Yreg = train_test_split(Xtotal, Ytotal, train_size=split_ratio1)
    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xreg, Yreg, train_size=split_ratio2)

    # train models
    models = []
    denoms = []
    features_list = []

    for mdl_idx in range(7): # modsel = 7
        if mdl_idx < 3: 
            qt = 0.25 * (mdl_idx + 1)
            mdl = QuantileRegressor(quantile=qt)
            features = np.random.choice(dim, dim // 10)
        elif mdl_idx in [3, 4]:
            mdl = LinearRegression()
            features = np.random.choice(dim, dim // 10)
        elif mdl_idx in [5, 6]:
            mdl = LinearRegression()
            features = np.random.choice(dim, dim // 5)

        mdl.fit(Xtrain[:, features], Ytrain)

        if mdl_idx < 3:
            denom = 1
        else:
            Xsigma, Ysigma = gen_data_Liang2024(setting, 100, sig, dim)
            if mdl_idx in [3, 4, 5]:
                # decrease n_estimators for runtime
                denom = LinearRegression()
            if mdl_idx in [6, 7, 8]:
                denom = SVR(kernel='rbf', gamma=0.1)
            denom.fit(Xsigma, np.absolute(Ysigma - mdl.predict(Xsigma[:, features])))

        models.append(mdl)
        denoms.append(denom)
        features_list.append(features)

    # select models
    Xmodsel_calib, Xmodsel_test, Ymodsel_calib, Ymodsel_test = train_test_split(Xmodsel, Ymodsel, train_size=1/2)

    try_selections = np.zeros((len(q_list), 7))

    for k in range(7):
        mdl = models[k]
        denom = denoms[k]
        features = features_list[k]

        calib_scores = 1000 * (Ymodsel_calib > 0) - mdl.predict(Xmodsel_calib[:, features]) / (1 if isinstance(denom, int) else denom.predict(Xmodsel_calib))
        test_scores = -mdl.predict(Xmodsel_test[:, features]) / (1 if isinstance(denom, int) else denom.predict(Xmodsel_test))
        pvals = np.zeros(len(Ymodsel_test))

        for j in range(len(Ymodsel_test)):
            pvals[j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)

        for qid, q in enumerate(q_list):
            try_selections[qid][k] = len(BH(pvals, q))

    # for each q, do selection based on the best model
    selections = []
    for qid, q in enumerate(q_list):
        max_sel = np.max(try_selections[qid])
        theta_j = np.random.choice([a for a in range(7) if try_selections[qid][a] == max_sel])

        mdl = models[theta_j]
        denom = denoms[theta_j]
        features = features_list[theta_j]

        calib_scores = 1000 * (Ycalib > 0) - mdl.predict(Xcalib[:, features]) / (1 if isinstance(denom, int) else denom.predict(Xcalib))
        test_scores = -mdl.predict(Xtest[:, features]) / (1 if isinstance(denom, int) else denom.predict(Xtest))
    
        pvals = np.zeros(len(Xtest))
        for j in range(len(Xtest)):
            pvals[j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)
    
        sel = BH(pvals, q)
        selections.append(sel)
    
    return selections

def LOO_train_modsel(mdl_idx, Xtotal, Ytotal, LOO_range, Xtest, strategy, m=None):
    """
    Conduct leave-one-out training and return predicted results.
    
    Parameters: 
    -- mdl_idx: model id to be fitted
    -- Xtotal, Ytotal: the total training data. For each time, we fit the model using the total data minus one sample.
    -- LOO_range: the indices in (Xtotal, Ytotal) that is left out (one at a time)
    -- Xtest: Samples where we predict Y using the LOO-trained models.
    -- strategy: Strategy to use when fitting the model
    -- m: length of testing (unlabelled data)

    Returns:
    -- pred_test: (ntest, ntotal)
       pred_test[i][j] is the predicted results of X_i (in Xtest) divided by denom using the j-th LOO model
    -- pred_loo: (ntotal)
       pred_loo[i] is the predicted results of the left-out sample divided by denom using the i-th LOO model.
    """
    # get the model
    if mdl_idx < 3: 
        qt = 0.25 * (mdl_idx + 1)
        mdl = QuantileRegressor(quantile=qt)
        features = np.random.choice(dim, dim // 10)
    elif mdl_idx in [3, 4]:
        mdl = LinearRegression()
        features = np.random.choice(dim, dim // 10)
    elif mdl_idx in [5, 6]:
        mdl = LinearRegression()
        features = np.random.choice(dim, dim // 5)

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
            rem_zero_idx = np.setdiff1d(np.where(Ytrain == 0)[0], set(sampled_zero_idx))
            
            if len(sampled_zero_idx) < m-1:
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(Xtrain[rem_zero_idx])
                _, nearest_idx = nbrs.kneighbors(Xtrain[sampled_one_idx])
                remove_idx = np.concatenate((sampled_zero_idx, rem_zero_idx[nearest_idx.flatten()]))
            else:
                remove_idx = sampled_zero_idx

            Xtrain, Ytrain = np.delete(Xtrain, remove_idx, axis=0), np.delete(Ytrain, remove_idx)

        if mdl_idx < 3:
            mdl.fit(Xtrain[:, features], Ytrain)
            denom = 1
        else:
            if mdl_idx in [3, 5]:
                Xtrain, Xsigma, Ytrain, Ysigma = train_test_split(Xtrain, Ytrain, train_size=0.7) # split for error model
                mdl.fit(Xtrain[:, features], Ytrain)
                # decrease n_estimators for runtime
                denom = LinearRegression()
                denom.fit(Xsigma, np.absolute(Ysigma - mdl.predict(Xsigma[:, features])))
            if mdl_idx in [4, 6]:
                mdl.fit(Xtrain[:, features], Ytrain)
                denom = 1

        if Xtest is not None:
            pred_test[:, ii] = mdl.predict(Xtest[:, features]) / (1 if isinstance(denom, int) else denom.predict(Xtest))
        X_loo = Xtotal[LOO_range[ii]].reshape((1, -1))
        pred_loo[ii] = mdl.predict(X_loo[:, features])[0] / (1 if isinstance(denom, int) else denom.predict(X_loo)[0]) # predict only the left out sample
        
    return pred_test, pred_loo

parser = argparse.ArgumentParser(description='Parser.')
parser.add_argument('nlabel', type=int)
parser.add_argument('ntest', type=int)
parser.add_argument('setting', type=int)  
parser.add_argument('seed', type=int)  
parser.add_argument('sigma', type=float)
parser.add_argument('dim', type=int)

args = parser.parse_args()

modelnum = 7 # FIXED HERE
setting = args.setting 
seedgroup = args.seed
nlabel = args.nlabel
ntest = args.ntest
sig = args.sigma
dim = args.dim

all_res = pd.DataFrame()
q_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

Nrep = 5

for seed in np.arange(Nrep * seedgroup, Nrep * (seedgroup+1)):
    random.seed(int(seed))
    np.random.seed(seed)

    print("Running seed "+str(seed)+" ...")
    
    Xlabel, Ylabel = gen_data_Liang2024(setting, nlabel, sig, dim)
    Xtest, Ytest = gen_data_Liang2024(setting, ntest, sig, dim) 
    Ylabel = 1 * (Ylabel > 0)
    Ytest = 1 * (Ytest > 0)

    """ Baseline 1: randomly picking one model + LOO 1.2/1.5/2e/2 """

    Xtotal = np.concatenate([Xlabel, Xtest])
    Ytotal = np.concatenate([1 * (Ylabel > 0), np.zeros(ntest)]) 

    ALL_PRED = []
    for k in range(modelnum):
        _, pred_all = LOO_train_modsel(k, Xtotal, Ytotal, np.arange(len(Xtotal)), None, strategy='oversample')
        ALL_PRED.append(pred_all)

    print('loo training complete...')

    baseline1_df = pd.DataFrame()

    baseline1_12_pvals, baseline1_15_pvals, baseline1_2_pvals = np.zeros(ntest), np.zeros(ntest), np.zeros(ntest)
    aux_size = np.zeros((ntest, len(q_list)))
    aux_avg_size = np.zeros((ntest, len(q_list)))

    model_idx = np.random.randint(0, modelnum) # random select a model

    # loo 1.2
    pred_all = ALL_PRED[model_idx]

    pred_labeled = pred_all[range(nlabel)]
    pred_test = pred_all[range(nlabel, nlabel+ntest)]

    calib_scores = 1000 * (Ylabel > 0) - pred_labeled
    test_scores = -pred_test
    for j in range(ntest):
        baseline1_12_pvals[j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1) 
        
    for qid, q in enumerate(q_list):
        Baseline1_12_sel = BH(baseline1_12_pvals, q) 
        Baseline1_2_sel = BH(baseline1_2_pvals, q)  
        Baseline1_12_FDP, Baseline1_12_power = eval(Ytest, Baseline1_12_sel, 0, np.inf) 

        baseline1_df = pd.concat((baseline1_df, pd.DataFrame({
            'Baseline1_12_FDP': [Baseline1_12_FDP],
            'Baseline1_12_power': [Baseline1_12_power] 
        })))

    print('loo 1.2 complete...') 


    """ Baseline 2: Data split for model selection - split labelled data + LOO 1.2/1.5/2e/2 """

    baseline2_df = pd.DataFrame()

    Xmodsel, Xlabel_0, Ymodsel, Ylabel_0 = train_test_split(Xlabel, Ylabel, train_size=0.5)
    Xmodsel_label, Xmodsel_test, Ymodsel_label, Ymodsel_test = train_test_split(Xmodsel, Ymodsel, train_size=0.5)

    Xtotal = np.concatenate([Xmodsel_label, Xmodsel_test])
    Ytotal = np.concatenate([1 * (Ymodsel_label > 0), np.zeros_like(Ymodsel_test)]) 

    model_selection_list = np.zeros((modelnum, len(q_list)))

    ALL_PRED_SEL = []

    for k in range(modelnum):
        pvals = np.zeros(len(Xmodsel_test))
        _, pred_all = LOO_train_modsel(k, Xtotal, Ytotal, np.arange(len(Xtotal)), None, strategy='oversample', m=len(Xmodsel_test))
        ALL_PRED_SEL.append(pred_all)

        pred_labeled = pred_all[range(len(Xmodsel_label))]
        pred_test = pred_all[range(len(Xmodsel_label), len(Xmodsel_label)+len(Xmodsel_test))]

        calib_scores = 1000 * (Ymodsel_label > 0) - pred_labeled
        test_scores = -pred_test
        for j in range(len(Xmodsel_test)):
            pvals[j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)
        for qid, q in enumerate(q_list):
            try_sel = BH(pvals, q)
            model_selection_list[k, qid] = len(try_sel)

    ALL_PRED_USE = []
    Xtotal_use = np.concatenate([Xlabel_0, Xtest])
    Ytotal_use = np.concatenate([1*(Ylabel_0>0), np.zeros(ntest)])

    # prepare for model selection at each fdr level 

    for k in range(modelnum): 
        _, pred_all = LOO_train_modsel(k, Xtotal_use, Ytotal_use, np.arange(len(Xtotal_use)), None, strategy='oversample', m=len(Xmodsel_test))
        ALL_PRED_USE.append(pred_all) 


    # loo 1.2
    max_modsel_sel = np.max(model_selection_list, axis=0) 

    for qid, q in enumerate(q_list): 
        theta_j = np.random.choice([a for a in range(modelnum) if model_selection_list[a, qid] == max_modsel_sel[qid]])

        baseline2_12_pvals, baseline2_15_pvals, baseline2_2_pvals = np.zeros(ntest), np.zeros(ntest), np.zeros(ntest)
        aux_size = np.zeros(ntest) # q-specific
    
        pred_all = ALL_PRED_USE[theta_j]

        pred_labeled = pred_all[range(len(Xlabel_0))]
        pred_test = pred_all[range(len(Xlabel_0), len(Xlabel_0)+ntest)]

        calib_scores = 1000 * (Ylabel_0 > 0) - pred_labeled
        test_scores = -pred_test
        for j in range(ntest):
            baseline2_12_pvals[j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)
        
        Baseline2_12_sel = BH(baseline2_12_pvals, q)
        # Baseline2_15_sel = BH(baseline2_15_pvals, q)
        # Baseline2_2_sel = BH(baseline2_2_pvals, q)  
        Baseline2_12_FDP, Baseline2_12_power = eval(Ytest, Baseline2_12_sel, 0, np.inf)
    
        
        baseline2_df = pd.concat((baseline2_df, pd.DataFrame({
            'Baseline2_12_FDP': [Baseline2_12_FDP],
            'Baseline2_12_power': [Baseline2_12_power]  
        })))

    print('random + loo complete...')


    """ Baseline 3: random selection + split conformal selection """
    
    baseline3_df = pd.DataFrame()

    model_idx = np.random.randint(0, modelnum) # random select a model
    baseline3_pvals = np.zeros((3, ntest))

    for r_idx, split_r in enumerate([0.25, 0.5, 0.75]):
        calib_scores, test_scores = Split_train_modsel(model_idx, Xlabel, Ylabel, split_r, Xtest)

        for j in range(ntest):
            baseline3_pvals[r_idx][j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)

    for qid, q in enumerate(q_list):
        Baseline3_25_sel = BH(baseline3_pvals[0], q)
        Baseline3_50_sel = BH(baseline3_pvals[1], q)
        Baseline3_75_sel = BH(baseline3_pvals[2], q)

        Baseline3_25_FDP, Baseline3_25_power = eval(Ytest, Baseline3_25_sel, 0, np.inf)
        Baseline3_50_FDP, Baseline3_50_power = eval(Ytest, Baseline3_50_sel, 0, np.inf)
        Baseline3_75_FDP, Baseline3_75_power = eval(Ytest, Baseline3_75_sel, 0, np.inf)

        baseline3_df = pd.concat((baseline3_df, pd.DataFrame({
            'Baseline3_25_FDP': [Baseline3_25_FDP],
            'Baseline3_25_power': [Baseline3_25_power],
            'Baseline3_50_FDP': [Baseline3_50_FDP],
            'Baseline3_50_power': [Baseline3_50_power],
            'Baseline3_75_FDP': [Baseline3_75_FDP],
            'Baseline3_75_power': [Baseline3_75_power]
        })))

    print('random baseline complete...')

    """ Baseline 4: split for model selection + split conformal selection """
            
    baseline4_df = pd.DataFrame()

    Baseline4_112_sel = Modsel_train_modsel(Xlabel, Ylabel, 0.25, 1/3, Xtest, q_list)
    Baseline4_121_sel = Modsel_train_modsel(Xlabel, Ylabel, 0.25, 2/3, Xtest, q_list)
    Baseline4_211_sel = Modsel_train_modsel(Xlabel, Ylabel, 0.50, 1/2, Xtest, q_list)
    Baseline4_111_sel = Modsel_train_modsel(Xlabel, Ylabel, 1/3, 1/2, Xtest, q_list)

    for qid, q in enumerate(q_list):
        Baseline4_112_FDP, Baseline4_112_power = eval(Ytest, Baseline4_112_sel[qid], 0, np.inf)
        Baseline4_121_FDP, Baseline4_121_power = eval(Ytest, Baseline4_121_sel[qid], 0, np.inf)
        Baseline4_211_FDP, Baseline4_211_power = eval(Ytest, Baseline4_211_sel[qid], 0, np.inf)
        Baseline4_111_FDP, Baseline4_111_power = eval(Ytest, Baseline4_111_sel[qid], 0, np.inf)

        baseline4_df = pd.concat((baseline4_df, pd.DataFrame({
            'Baseline4_112_FDP': [Baseline4_112_FDP],
            'Baseline4_112_power': [Baseline4_112_power],
            'Baseline4_121_FDP': [Baseline4_121_FDP],
            'Baseline4_121_power': [Baseline4_121_power],
            'Baseline4_211_FDP': [Baseline4_211_FDP],
            'Baseline4_211_power': [Baseline4_211_power],
            'Baseline4_111_FDP': [Baseline4_111_FDP],
            'Baseline4_111_power': [Baseline4_111_power]
        })))

    print('split baseline complete...')


    """ Alt 1 - model selection on full data + OptCS-Full """

    alt_df = pd.DataFrame()

    Xtotal = np.concatenate([Xlabel, Xtest])
    Ytotal = np.concatenate([1 * (Ylabel > 0), np.zeros(ntest)])

    idx = np.random.choice(len(Xtotal), size=ntest)
    X_sample, Y_sample = Xtotal[idx], Ytotal[idx]

    # first remove the zeros
    sampled_zero_idx = idx[np.where(Y_sample == 0)[0]]
    sampled_one_idx = idx[np.where(Y_sample == 1)[0]]
    rem_zero_idx = np.setdiff1d(np.where(Ytotal == 0)[0], set(sampled_zero_idx))

    if len(sampled_zero_idx) < ntest:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(Xtotal[rem_zero_idx])
        _, nearest_idx = nbrs.kneighbors(Xtotal[sampled_one_idx])
        remove_idx = np.concatenate((sampled_zero_idx, rem_zero_idx[nearest_idx.flatten()]))
    else: # = ntest
        remove_idx = sampled_zero_idx
    # remove_idx correspond to test
    train_idx = np.setdiff1d(np.arange(len(pred_all)), remove_idx)

    model_selection_list = np.zeros((modelnum, len(q_list)))
    for k in range(modelnum):
        pred_all = ALL_PRED[k]
        
        pred_labeled = pred_all[train_idx]
        pred_test = pred_all[remove_idx]

        calib_scores = 1000 * (Ytotal[train_idx] > 0) - pred_labeled
        test_scores = -pred_test

        pvals = np.zeros(ntest)
        for j in range(ntest):
            pvals[j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)
        for qid, q in enumerate(q_list):
            try_sel = BH(pvals, q)
            model_selection_list[k, qid] = len(try_sel)

    for qid, q in enumerate(q_list):
        alt_pvals = np.zeros(ntest)

        max_modsel_sel = np.max(model_selection_list[:, qid])
        theta_j = np.random.choice([a for a in range(modelnum) if model_selection_list[a, qid] == max_modsel_sel])

        pred_all = ALL_PRED[theta_j]

        pred_labeled = pred_all[range(nlabel)]
        pred_test = pred_all[range(nlabel, nlabel+ntest)]

        calib_scores = 1000 * (Ylabel > 0) - pred_labeled
        test_scores = -pred_test
        for j in range(ntest):
            alt_pvals[j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)

        alt_sel = BH(alt_pvals, q)
        alt_FDP, alt_power = eval(Ytest, alt_sel, 0, np.inf)

        alt_df = pd.concat((alt_df, pd.DataFrame({
            'alt_FDP': [alt_FDP],
            'alt_power': [alt_power]
        })))

    print('loo sel+training complete...')



    """ OptCS """

    optcs_df = pd.DataFrame()

    Xtotal = np.concatenate([Xlabel, Xtest])
    Ytotal = np.concatenate([1 * (Ylabel > 0), np.zeros(ntest)])

    for qid, q in enumerate(q_list):
        pvals = []
        pvals_rand = []
        max_aux_sel_sizes = []
        max_aux_sel_sizes_9 = []
        for j in range(ntest):
            aux_sel_sizes = []
            aux_sel_sizes_9 = [] 
            for k in range(modelnum):
                pred_all = ALL_PRED[k]

                pred_labeled = pred_all[range(nlabel)]
                pred_test = pred_all[range(nlabel, nlabel+ntest)]

                calib_scores = 1000 * (Ylabel > 0) - pred_labeled
                test_scores = -pred_test

                aux_pvals = np.zeros(ntest)
                for l in range(ntest):
                    if l != j:
                        aux_pvals[l] = (np.sum(calib_scores <= test_scores[l]) + (test_scores[j] <= test_scores[l])) / (len(calib_scores) + 1)
                aux_sel = BH(aux_pvals, q)
                aux_sel_sizes.append(len(aux_sel))
                aux_sel_9 = BH(aux_pvals, q * 0.9)
                aux_sel_sizes_9.append(len(aux_sel_9))
            
            # determine the best model when considering X_{n+j}
            max_aux_sel = max(aux for aux in aux_sel_sizes)
            theta_j = np.random.choice([a for a in range(modelnum) if aux_sel_sizes[a] == max_aux_sel])
            max_aux_sel_sizes.append(max_aux_sel)
            max_aux_sel_sizes_9.append(max(aux_sel_sizes_9))

            # now use theta_j to get the p-value
            pred_all = ALL_PRED[theta_j]

            pred_labeled = pred_all[range(nlabel)]
            pred_test = pred_all[range(nlabel, nlabel+ntest)]

            calib_scores = 1000 * (Ylabel > 0) - pred_labeled
            test_scores = -pred_test
            p_j = (1 + np.sum(calib_scores <= test_scores[j])) / (len(calib_scores) + 1)
            # randomized
            p_j_rand = np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1) / (len(calib_scores) + 1)
            
            pvals.append(p_j)
            pvals_rand.append(p_j_rand)

        max_aux_sel_sizes = np.array(max_aux_sel_sizes)
        max_aux_sel_sizes_9 = np.array(max_aux_sel_sizes_9)

        pvals = np.array(pvals)

        evalues = (pvals <= q * max_aux_sel_sizes / ntest) / (q * max_aux_sel_sizes / ntest) + 1e-8
        optcs_dtm_sel = eBH(evalues, q)
        optcs_dtm_FDP, optcs_dtm_power = eval(Ytest, optcs_dtm_sel, 0, np.inf)

        optcs_hete_sel = eBH(evalues / np.random.uniform(0,1,size=ntest), q)
        optcs_hete_FDP, optcs_hete_power = eval(Ytest, optcs_hete_sel, 0, np.inf)

        optcs_homo_sel = eBH(evalues / np.random.uniform(0,1,size=1)[0], q)
        optcs_homo_FDP, optcs_homo_power = eval(Ytest, optcs_homo_sel, 0, np.inf)

        evalues_9 = (pvals <= q * max_aux_sel_sizes * 0.9 / ntest) / (q * max_aux_sel_sizes * 0.9 / ntest)
        evalues_9[evalues_9 == 0] = 1e-9

        optcs_dtm_9_sel = eBH(evalues_9, q)
        optcs_dtm_9_FDP, optcs_dtm_9_power = eval(Ytest, optcs_dtm_9_sel, 0, np.inf)

        optcs_hete_9_sel = eBH(evalues_9 / np.random.uniform(0,1,size=ntest), q)
        optcs_hete_9_FDP, optcs_hete_9_power = eval(Ytest, optcs_hete_9_sel, 0, np.inf)

        optcs_homo_9_sel = eBH(evalues_9 / np.random.uniform(0,1,size=1)[0], q)
        optcs_homo_9_FDP, optcs_homo_9_power = eval(Ytest, optcs_homo_9_sel, 0, np.inf)

        evalues_9_alt = (pvals <= q * max_aux_sel_sizes_9 / ntest) / (q * max_aux_sel_sizes_9 / ntest)
        evalues_9_alt[evalues_9_alt == 0] = 1e-9

        optcs_dtm_9_alt_sel = eBH(evalues_9_alt, q)
        optcs_dtm_9_alt_FDP, optcs_dtm_9_alt_power = eval(Ytest, optcs_dtm_9_sel, 0, np.inf)

        optcs_hete_9_alt_sel = eBH(evalues_9_alt / np.random.uniform(0,1,size=ntest), q)
        optcs_hete_9_alt_FDP, optcs_hete_9_alt_power = eval(Ytest, optcs_hete_9_sel, 0, np.inf)

        optcs_homo_9_alt_sel = eBH(evalues_9_alt / np.random.uniform(0,1,size=1)[0], q)
        optcs_homo_9_alt_FDP, optcs_homo_9_alt_power = eval(Ytest, optcs_homo_9_sel, 0, np.inf)

        optcs_df = pd.concat((optcs_df, pd.DataFrame({
            'OptCS_dtm_FDP': [optcs_dtm_FDP],
            'OptCS_dtm_power': [optcs_dtm_power],
            'OptCS_hete_FDP': [optcs_hete_FDP],
            'OptCS_hete_power': [optcs_hete_power],
            'OptCS_homo_FDP': [optcs_homo_FDP],
            'OptCS_homo_power': [optcs_homo_power],
            'OptCS_dtm_0.9_FDP': [optcs_dtm_9_FDP],
            'OptCS_dtm_0.9_power': [optcs_dtm_9_power],
            'OptCS_hete_0.9_FDP': [optcs_hete_9_FDP],
            'OptCS_hete_0.9_power': [optcs_hete_9_power],
            'OptCS_homo_0.9_FDP': [optcs_homo_9_FDP],
            'OptCS_homo_0.9_power': [optcs_homo_9_power],
            'OptCS_dtm_0.9_alt_FDP': [optcs_dtm_9_alt_FDP],
            'OptCS_dtm_0.9_alt_power': [optcs_dtm_9_alt_power],
            'OptCS_hete_0.9_alt_FDP': [optcs_hete_9_alt_FDP],
            'OptCS_hete_0.9_alt_power': [optcs_hete_9_alt_power],
            'OptCS_homo_0.9_alt_FDP': [optcs_homo_9_alt_FDP],
            'OptCS_homo_0.9_alt_power': [optcs_homo_9_alt_power],
            'q': [q],
            'setting': [setting],
            'dim': [dim],
            'seed': [seed],
            'sigma': [sig],
            'ntest': [ntest],
            'nlabel': [nlabel]
        })))

    print('optcs complete...')

    this_res = pd.concat([baseline1_df, baseline2_df, baseline3_df, baseline4_df, alt_df, optcs_df], axis=1)

    all_res = pd.concat([all_res, this_res])


if not os.path.exists('results'):
    os.makedirs('results')

all_res.to_csv(os.path.join('results', f"choosescore_LOO_Liang, setting={setting}, dim={dim}, sigma={sig}, seedgroup={seedgroup}, ntest={ntest}, nlabel={nlabel}.csv"))