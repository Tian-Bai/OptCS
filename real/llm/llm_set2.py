import os
import pandas as pd
import numpy as np
import random
import argparse

from xgboost import XGBRFClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import NearestNeighbors
from utility import BH, eBH, eval
from tqdm import tqdm

def get_model(mdl_idx, features_list):
    if mdl_idx < 5:
        return RandomForestClassifier(max_depth=10, random_state=2024), features_list[mdl_idx % 5]
    if 5 <= mdl_idx and mdl_idx < 10:
        return LogisticRegression(random_state=0, solver='liblinear'), features_list[mdl_idx % 5]
    if mdl_idx >= 10:
        return XGBRFClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.2), features_list[mdl_idx % 5]

def Modsel_train_modsel(Xlabel, Ylabel, features_list, split_ratio1, split_ratio2, Xtest, q_list):
    Xmodsel, Xreg, Ymodsel, Yreg = train_test_split(Xlabel, Ylabel, train_size=split_ratio1)
    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xreg, Yreg, train_size=split_ratio2)

    models = []

    for mdl_idx in range(modelnum):
        mdl, features = get_model(mdl_idx, features_list)
        mdl.fit(Xtrain[:, features], Ytrain)
        models.append(mdl)
    
    Xmodsel_calib, Xmodsel_test, Ymodsel_calib, Ymodsel_test = train_test_split(Xmodsel, Ymodsel, train_size=1/2)
    try_selections = np.zeros((len(q_list), modelnum))

    for k in range(modelnum):
        mdl = models[k]
        features = features_list[k % 5]

        calib_scores = 1000 * (Ymodsel_calib > 0) - mdl.predict_proba(Xmodsel_calib[:, features])[:, 1]
        test_scores = -mdl.predict_proba(Xmodsel_test[:, features])[:, 1]
        pvals = np.zeros(len(Ymodsel_test))

        for j in range(len(Ymodsel_test)):
            pvals[j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)

        for qid, q in enumerate(q_list):
            try_selections[qid][k] = len(BH(pvals, q))

    selections = []
    for qid, q in enumerate(q_list):
        max_sel = np.max(try_selections[qid])
        theta_j = np.random.choice([a for a in range(modelnum) if try_selections[qid][a] == max_sel])

        mdl = models[theta_j]
        features = features_list[theta_j % 5]

        calib_scores = 1000 * (Ycalib > 0) - mdl.predict_proba(Xcalib[:, features])[:, 1]
        test_scores = -mdl.predict_proba(Xtest[:, features])[:, 1]
    
        baseline2_pvals = np.zeros(len(Xtest))
        for j in range(len(Xtest)):
            baseline2_pvals[j] = (np.sum(calib_scores < test_scores[j]) + (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)
    
        sel = BH(baseline2_pvals, q)
        selections.append(sel)

    return selections

def LOO_train_modsel(mdl_idx, features_list, Xtotal, Ytotal, LOO_range, Xtest, strategy, m=None):
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
       pred_test[i][j] is the predicted results of X_i (in Xtest) using the j-th LOO model
    -- pred_loo: (ntotal)
       pred_loo[i] is the predicted results of the left-out sample using the i-th LOO model.
    """
    # get the model
    mdl, features = get_model(mdl_idx, features_list)

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
            rem_zero_idx = np.setdiff1d(np.where(Ytrain == 0)[0], sampled_zero_idx)
            
            if len(sampled_zero_idx) < m-1:
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(Xtrain[rem_zero_idx])
                _, nearest_idx = nbrs.kneighbors(Xtrain[sampled_one_idx])
                remove_idx = np.concatenate((sampled_zero_idx, rem_zero_idx[nearest_idx.flatten()]))
            else:
                remove_idx = sampled_zero_idx

            Xtrain, Ytrain = np.delete(Xtrain, remove_idx, axis=0), np.delete(Ytrain, remove_idx)

        mdl.fit(Xtrain[:, features], Ytrain)

        if Xtest is not None:
            pred_test[:, ii] = mdl.predict_proba(Xtest[:, features])[:, 1]
        X_loo = Xtotal[LOO_range[ii]].reshape((1, -1))
        pred_loo[ii] = mdl.predict_proba(X_loo[:, features])[0, 1]
        
    return pred_test, pred_loo

parser = argparse.ArgumentParser()
parser.add_argument('--seedgroup', type=int, required=True)
parser.add_argument('--repN', type=int, default=500)
args = parser.parse_args()

repN = args.repN
seedgroup = args.seedgroup

q_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
modelnum = 12

fdp_rep = []
power_rep = []

all_res = pd.DataFrame()

save_df = pd.read_csv("saved_feature_and_label.csv")
X, Y = save_df.drop(columns='Response').to_numpy(), save_df['Response'].to_numpy()

print(X.shape, Y.shape)

for rep_idx in tqdm(range(repN * seedgroup, repN * (seedgroup + 1))):
    random.seed(rep_idx)
    np.random.seed(rep_idx)

    # use one group of features, or all features
    features_list = [np.arange(3), np.arange(3, 6), np.arange(6, 9), np.arange(9, 12), np.arange(12)]

    Xlabel, Xtest, Ylabel, Ytest = train_test_split(X, Y, train_size=4/5)

    """ Naive """

    naive_df = pd.DataFrame()
    indiv_df = pd.DataFrame()

    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xlabel, Ylabel, train_size=5/8)

    models = []
    for mdl_idx in range(modelnum):
        mdl, features = get_model(mdl_idx, features_list)
        mdl.fit(Xtrain[:, features], Ytrain)
        models.append(mdl)

    for qid, q in enumerate(q_list):
        naive_selection_list = []
        indiv_FDPs, indiv_powers = [], []

        indiv_q_df = pd.DataFrame()

        for k in range(modelnum):
            mdl = models[k]
            features = features_list[k % 5]

            calib_scores = 1000 * (Ycalib > 0) - mdl.predict_proba(Xcalib[:, features])[:, 1]
            test_scores = -mdl.predict_proba(Xtest[:, features])[:, 1]

            pvals = np.zeros(len(Ytest))
            for j in range(len(Ytest)):
                pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)
            sel = BH(pvals, q) # test using that specific nominal level
            if True:
                indiv_FDP, indiv_power = eval(Ytest, sel, 0, np.inf)

                indiv_q_df = pd.concat((indiv_q_df, pd.DataFrame({
                    f'Indiv_{k}_FDP': [indiv_FDP],
                    f'Indiv_{k}_power': [indiv_power]
                })), axis=1)
            naive_selection_list.append(sel)

        indiv_df = pd.concat((indiv_df, indiv_q_df))

        # pick one model that leads to the largest selection set (and break ties randomly)
        max_naive_sel = max(len(sel) for sel in naive_selection_list)
        naive_sel = naive_selection_list[np.random.choice([k for k in range(len(naive_selection_list)) if len(naive_selection_list[k]) == max_naive_sel])]
        naive_FDP, naive_power = eval(Ytest, naive_sel, 0, np.inf)

        naive_df = pd.concat((naive_df, pd.DataFrame({
            'Naive_FDP': [naive_FDP],
            'Naive_power': [naive_power]
        })))

    print('naive method complete...')

    """ Baseline 1: randomly select a classifier + vanilla conformal selection """

    baseline1_df = pd.DataFrame()

    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xlabel, Ylabel, train_size=5/8)

    mdl_idx = np.random.choice(modelnum)
    mdl, features = get_model(mdl_idx, features_list)
    
    mdl.fit(Xtrain[:, features], Ytrain)
    
    Ycalib_pred = mdl.predict_proba(Xcalib[:, features])[:, 1]
    Ytest_pred = mdl.predict_proba(Xtest[:, features])[:, 1]

    calib_scores = 1000 * (Ycalib > 0) - Ycalib_pred
    test_scores = -Ytest_pred

    baseline1_pvals = np.zeros(len(Xtest))
    for j in range(len(Xtest)):
        baseline1_pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)

    for qid, q in enumerate(q_list):
        baseline1_sel = BH(baseline1_pvals, q)
        baseline1_FDP, baseline1_power = eval(Ytest, baseline1_sel, 0, np.inf)

        baseline1_df = pd.concat((baseline1_df, pd.DataFrame({
            'Baseline1_FDP': [baseline1_FDP],
            'Baseline1_power': [baseline1_power]
        })))

    print("baseline1 complete...")

    """ Baseline 2: select a classifier by partitioning + vanilla conformal selection """

    baseline2_df = pd.DataFrame()

    Baseline2_112_sel = Modsel_train_modsel(Xlabel, Ylabel, features_list, 0.25, 1/3, Xtest, q_list)
    Baseline2_121_sel = Modsel_train_modsel(Xlabel, Ylabel, features_list, 0.25, 2/3, Xtest, q_list)
    Baseline2_211_sel = Modsel_train_modsel(Xlabel, Ylabel, features_list, 0.50, 1/2, Xtest, q_list)
    Baseline2_111_sel = Modsel_train_modsel(Xlabel, Ylabel, features_list, 1/3, 1/2, Xtest, q_list)

    for qid, q in enumerate(q_list):
        Baseline2_112_FDP, Baseline2_112_power = eval(Ytest, Baseline2_112_sel[qid], 0, np.inf)
        Baseline2_121_FDP, Baseline2_121_power = eval(Ytest, Baseline2_121_sel[qid], 0, np.inf)
        Baseline2_211_FDP, Baseline2_211_power = eval(Ytest, Baseline2_211_sel[qid], 0, np.inf)
        Baseline2_111_FDP, Baseline2_111_power = eval(Ytest, Baseline2_111_sel[qid], 0, np.inf)

        baseline2_df = pd.concat((baseline2_df, pd.DataFrame({
            'Baseline2_112_FDP': [Baseline2_112_FDP],
            'Baseline2_112_power': [Baseline2_112_power],
            'Baseline2_121_FDP': [Baseline2_121_FDP],
            'Baseline2_121_power': [Baseline2_121_power],
            'Baseline2_211_FDP': [Baseline2_211_FDP],
            'Baseline2_211_power': [Baseline2_211_power],
            'Baseline2_111_FDP': [Baseline2_111_FDP],
            'Baseline2_111_power': [Baseline2_111_power]
        })))

    print('baseline2 complete...')

    """ OptCS-Modsel-Full """

    optcs_df = pd.DataFrame()

    ntest = len(Xtest)
    nlabel = len(Xlabel)
    Xtotal = np.concatenate([Xlabel, Xtest])
    Ytotal = np.concatenate([1 * (Ylabel > 0), np.zeros(ntest)]) 

    ALL_PRED = []
    for k in range(modelnum):
        _, pred_all = LOO_train_modsel(k, features_list, Xtotal, Ytotal, np.arange(len(Xtotal)), None, strategy='oversample')
        ALL_PRED.append(pred_all)

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
            p_j_rand = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)
            # use_rand?
            # p_j = p_j_rand
            
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
            'seed': [rep_idx],
            'ntest': [ntest],
            'nlabel': [nlabel],
        })))

    print("optcs complete...")

    this_res = pd.concat([baseline1_df, baseline2_df, optcs_df, naive_df, indiv_df], axis=1)

    all_res = pd.concat([all_res, this_res])

if not os.path.exists('results'):
    os.makedirs('results', exist_ok=True)

all_res.to_csv(os.path.join('results', f"cxr, setup=2, repN={repN}, seedgroup={seedgroup}.csv"))