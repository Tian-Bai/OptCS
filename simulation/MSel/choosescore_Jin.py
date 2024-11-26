import numpy as np
import pandas as pd 
import random
import sys
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from quantile_forest import RandomForestQuantileRegressor
from sklearn.svm import SVR
from utils import gen_data_Jin2023, eval, BH, eBH
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import argparse

parser = argparse.ArgumentParser(description='Parser.')
parser.add_argument('ntrain', type=int)
parser.add_argument('ncalib', type=int)
parser.add_argument('ntest', type=int)
parser.add_argument('setting', type=int)
parser.add_argument('sigma', type=float)
parser.add_argument('dim', type=int)
parser.add_argument('report_indiv', type=int, default=0) # whether to report individual model/score performances
parser.add_argument('Nrep', type=int)
parser.add_argument('seedgroup', type=int)

args = parser.parse_args()

modelnum = 24

setting = args.setting
ntrain = args.ntrain
ncalib = args.ncalib # added this to the argument, since the performance of baseline2 might be sensitive to this term.
ntest = args.ntest # should not be too large, since the time complexity is quadratic in this term.
sig = args.sigma
dim = args.dim # feature. for YJ, keep it as 20 for now.
report_indiv = bool(args.report_indiv)
Nrep = args.Nrep
seedgroup = args.seedgroup

q_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

all_res = pd.DataFrame()

for i_itr in range(Nrep * seedgroup, Nrep * (seedgroup+1)):
    random.seed(i_itr)
    np.random.seed(i_itr)

    print("Running seed "+str(i_itr)+" ...")

    Xtrain, Ytrain = gen_data_Jin2023(setting, ntrain, sig, dim)
    Xcalib, Ycalib = gen_data_Jin2023(setting, ncalib, sig, dim)
    Xtest, Ytest = gen_data_Jin2023(setting, ntest, sig, dim)

    model_list = []
    denom_list = [] # denominator in the nonconformity score
    
    # 18 conditional quantile regressors: rf, gb
    for qq in range(1, 10):
        qt = qq / 10 # quantile level
        rf_qt = RandomForestQuantileRegressor(max_depth=5, default_quantiles=qt, random_state=0)
        gb_qt = GradientBoostingRegressor(loss='quantile', alpha=qt, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)
        rf_qt.fit(Xtrain, Ytrain)
        gb_qt.fit(Xtrain, Ytrain)
        model_list.append(rf_qt)
        model_list.append(gb_qt)
        denom_list += [1, 1]

    # 1 conditional mean regressors: rf, paired with 6 sigma estimators: (None, gb, rf, svr, lasso, ridge)
    rf = RandomForestRegressor(max_depth=5, random_state=0)
    rf.fit(Xtrain, Ytrain)

    # combine them with some choices of sigma
    # generate more data for fitting sigma
    Xsigma, Ysigma = gen_data_Jin2023(setting, 100, sig, dim)
    gb_sigma = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)
    rf_sigma = RandomForestRegressor(max_depth=5, random_state=0)
    svr_sigma = SVR(kernel='rbf', gamma=0.1)
    lasso_sigma = Lasso(alpha=1)
    ridge_sigma = Ridge(alpha=1)

    gb_sigma.fit(Xsigma, np.absolute(Ysigma - rf.predict(Xsigma)))
    rf_sigma.fit(Xsigma, np.absolute(Ysigma - rf.predict(Xsigma)))
    svr_sigma.fit(Xsigma, np.absolute(Ysigma - rf.predict(Xsigma)))
    lasso_sigma.fit(Xsigma, np.absolute(Ysigma - rf.predict(Xsigma)))
    ridge_sigma.fit(Xsigma, np.absolute(Ysigma - rf.predict(Xsigma)))

    model_list += [rf] * 6
    denom_list += [1, gb_sigma, rf_sigma, svr_sigma, lasso_sigma, ridge_sigma]

    ''' Baseline 1: randomly picking one model '''
    
    baseline1_df = pd.DataFrame()

    model_idx = np.random.randint(0, modelnum)
    reg = model_list[model_idx]
    denom = denom_list[model_idx]

    calib_scores = 1000 * (Ycalib > 0) - reg.predict(Xcalib) / (1 if isinstance(denom, int) else denom.predict(Xcalib))
    test_scores = -reg.predict(Xtest) / (1 if isinstance(denom, int) else denom.predict(Xtest))

    baseline1_pvals = np.zeros(ntest)
    for j in range(ntest):
        baseline1_pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)
    
    # evaluate
    for qid, q in enumerate(q_list):
        baseline1_sel = BH(baseline1_pvals, q)
        Baseline1_FDP, Baseline1_power = eval(Ytest, baseline1_sel, 0, np.inf) 

        baseline1_df = pd.concat((baseline1_df, pd.DataFrame({
            'Baseline1_FDP': [Baseline1_FDP],
            'Baseline1_power': [Baseline1_power] 
        })))

    print('baseline1 complete...')

    ''' Baseline 2: Data split for model selection - split calibration data '''

    baseline2_df = pd.DataFrame()

    # for instance, select the model/score that lead to the most rejections
    Xmodsel, Xcalib_0, Ymodsel, Ycalib_0 = train_test_split(Xcalib, Ycalib, train_size=0.5) 
    Xmodelsel_calib, Xmodsel_test, Ymodsel_calib, Ymodsel_test = train_test_split(Xmodsel, Ymodsel, train_size=0.5) 

    # try the selection performance on (Xmodsel, Ymodsel)
    modsel_selection_list = np.zeros((len(q_list), modelnum))
    for k in range(modelnum):
        reg = model_list[k]
        denom = denom_list[k]

        calib_scores_modsel = 1000 * (Ymodsel_calib > 0) - reg.predict(Xmodelsel_calib) / (1 if isinstance(denom, int) else denom.predict(Xmodelsel_calib))
        test_scores_modsel = -reg.predict(Xmodsel_test) / (1 if isinstance(denom, int) else denom.predict(Xmodsel_test))
        pvals = np.zeros(len(Xmodsel_test))
        for j in range(len(Xmodsel_test)):
            pvals[j] = (np.sum(calib_scores_modsel < test_scores_modsel[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores_modsel == test_scores_modsel[j]) + 1)) / (len(calib_scores_modsel) + 1)
        for qid, q in enumerate(q_list):
            try_sel = BH(pvals, q)
            modsel_selection_list[qid, k] = len(try_sel)

    for qid, q in enumerate(q_list):
        max_modsel_sel = max(modsel_selection_list[qid])
        theta_j = np.random.choice([a for a in range(modelnum) if modsel_selection_list[qid, a] == max_modsel_sel])

        reg = model_list[theta_j]
        denom = denom_list[theta_j]

        calib_scores = 1000 * (Ycalib_0 > 0) - reg.predict(Xcalib_0) / (1 if isinstance(denom, int) else denom.predict(Xcalib_0))
        test_scores = -reg.predict(Xtest) / (1 if isinstance(denom, int) else denom.predict(Xtest))

        baseline2_pvals = np.zeros(ntest)
        for j in range(ntest):
            baseline2_pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)

        baseline2_sel = BH(baseline2_pvals, q)
        Baseline2_FDP, Baseline2_power = eval(Ytest, baseline2_sel, 0, np.inf) 

        baseline2_df = pd.concat((baseline2_df, pd.DataFrame({
            'Baseline2_FDP': [Baseline2_FDP],
            'Baseline2_power': [Baseline2_power] 
        })))

    print('baseline2 complete...')

    ''' Naive method: no correction '''
    
    naive_df = pd.DataFrame()
    indiv_df = pd.DataFrame()

    for qid, q in enumerate(q_list):
        naive_selection_list = []
        indiv_FDPs, indiv_powers = [], []

        indiv_q_df = pd.DataFrame()

        for k in range(modelnum):
            reg = model_list[k]
            denom = denom_list[k]

            calib_scores = 1000 * (Ycalib > 0) - reg.predict(Xcalib) / (1 if isinstance(denom, int) else denom.predict(Xcalib))
            test_scores = -reg.predict(Xtest) / (1 if isinstance(denom, int) else denom.predict(Xtest))

            pvals = np.zeros(ntest)
            for j in range(ntest):
                pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)
            sel = BH(pvals, q) # test using that specific nominal level
            if report_indiv:
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

    ''' Our purposed method OptCS '''

    OptCS_df = pd.DataFrame()

    for qid, q in enumerate(q_list):
        pvals = []
        max_aux_sel_sizes = []

        for j in range(ntest):
            aux_sel_sizes = []
            for k in range(modelnum):
                reg = model_list[k]
                denom = denom_list[k]

                calib_scores = 1000 * (Ycalib > 0) - reg.predict(Xcalib) / (1 if isinstance(denom, int) else denom.predict(Xcalib))
                test_scores = -reg.predict(Xtest) / (1 if isinstance(denom, int) else denom.predict(Xtest))

                aux_pvals = np.zeros(ntest)
                for l in range(ntest):
                    if l != j:
                        aux_pvals[l] = (np.sum(calib_scores <= test_scores[l]) + (test_scores[j] <= test_scores[l])) / (len(calib_scores) + 1)
                aux_sel = BH(aux_pvals, q)
                aux_sel_sizes.append(len(aux_sel))
            
            # determine the best model when considering X_{n+j}
            max_aux_sel = max(aux for aux in aux_sel_sizes)
            theta_j = np.random.choice([a for a in range(modelnum) if aux_sel_sizes[a] == max_aux_sel])
            max_aux_sel_sizes.append(max_aux_sel)

            # now use theta_j to get the p-value
            calib_scores = 1000 * (Ycalib > 0) - model_list[theta_j].predict(Xcalib) / (1 if isinstance(denom_list[theta_j], int) else denom_list[theta_j].predict(Xcalib))
            test_scores = -model_list[theta_j].predict(Xtest) / (1 if isinstance(denom_list[theta_j], int) else denom_list[theta_j].predict(Xtest))
            p_j = (1 + np.sum(calib_scores < test_scores[j])) / (len(calib_scores) + 1)
            pvals.append(p_j)
        
        max_aux_sel_sizes = np.array(max_aux_sel_sizes)
        pvals = np.array(pvals)

        # get the e-values
        evalues = (pvals <= q * max_aux_sel_sizes / ntest) / (q * max_aux_sel_sizes / ntest)
        evalues[evalues == 0] = 1e-9

        # deterministic pruning
        optcs_dtm_sel = eBH(evalues, q)
        optcs_dtm_FDP, optcs_dtm_power = eval(Ytest, optcs_dtm_sel, 0, np.inf)

        # heterogeneous pruning
        optcs_hete_sel = eBH(evalues / np.random.uniform(0,1,size=ntest), q)
        optcs_hete_FDP, optcs_hete_power = eval(Ytest, optcs_hete_sel, 0, np.inf)

        # homogeneous pruning
        optcs_homo_sel = eBH(evalues / np.random.uniform(0,1,size=1)[0], q)
        optcs_homo_FDP, optcs_homo_power = eval(Ytest, optcs_homo_sel, 0, np.inf)

        # get the e-values with beta = 0.9
        beta = 0.9
        evalues_9 = (pvals <= q * max_aux_sel_sizes * beta / ntest) / (q * max_aux_sel_sizes * beta / ntest)
        evalues_9[evalues_9 == 0] = 1e-9

        # deterministic pruning
        optcs_dtm_9_sel = eBH(evalues_9, q)
        optcs_dtm_9_FDP, optcs_dtm_9_power = eval(Ytest, optcs_dtm_9_sel, 0, np.inf)

        # heterogeneous pruning
        optcs_hete_9_sel = eBH(evalues_9 / np.random.uniform(0,1,size=ntest), q)
        optcs_hete_9_FDP, optcs_hete_9_power = eval(Ytest, optcs_hete_9_sel, 0, np.inf)

        # homogeneous pruning
        optcs_homo_9_sel = eBH(evalues_9 / np.random.uniform(0,1,size=1)[0], q)
        optcs_homo_9_FDP, optcs_homo_9_power = eval(Ytest, optcs_homo_9_sel, 0, np.inf)

        OptCS_df = pd.concat((OptCS_df, pd.DataFrame({
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
            'q': [q],
            'setting': [setting],
            'dim': [dim],
            'seed': [i_itr],
            'sigma': [sig],
            'ntest': [ntest],
            'ncalib': [ncalib],
            'ntrain': [ntrain]
        })))

    print('optcs complete...')

    ''' Baseline 3: Data split for model selection - split training data '''

    baseline3_df = pd.DataFrame()

    # refit the models using less data
    Xmodsel, Xtrain_0, Ymodsel, Ytrain_0 = train_test_split(Xtrain, Ytrain, train_size=0.5) 
    Xmodelsel_calib, Xmodsel_test, Ymodsel_calib, Ymodsel_test = train_test_split(Xmodsel, Ymodsel, train_size=0.5) 

    model_list = []
    denom_list = [] # denominator in the nonconformity score
    
    # 18 conditional quantile regressors: rf, gb
    for qq in range(1, 10):
        qt = qq / 10 # quantile level
        rf_qt = RandomForestQuantileRegressor(max_depth=5, default_quantiles=qt, random_state=0)
        gb_qt = GradientBoostingRegressor(loss='quantile', alpha=qt, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)
        rf_qt.fit(Xtrain_0, Ytrain_0)
        gb_qt.fit(Xtrain_0, Ytrain_0)
        model_list.append(rf_qt)
        model_list.append(gb_qt)
        denom_list += [1, 1]

    # 1 conditional mean regressors: rf, paired with 6 sigma estimators: (None, gb, rf, svr, lasso, ridge)
    rf = RandomForestRegressor(max_depth=5, random_state=0)
    rf.fit(Xtrain_0, Ytrain_0)

    # combine them with some choices of sigma
    # generate more data for fitting sigma
    Xsigma, Ysigma = gen_data_Jin2023(setting, 100, sig, dim)
    gb_sigma = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)
    rf_sigma = RandomForestRegressor(max_depth=5, random_state=0)
    svr_sigma = SVR(kernel='rbf', gamma=0.1)
    lasso_sigma = Lasso(alpha=1)
    ridge_sigma = Ridge(alpha=1)

    gb_sigma.fit(Xsigma, np.absolute(Ysigma - rf.predict(Xsigma)))
    rf_sigma.fit(Xsigma, np.absolute(Ysigma - rf.predict(Xsigma)))
    svr_sigma.fit(Xsigma, np.absolute(Ysigma - rf.predict(Xsigma)))
    lasso_sigma.fit(Xsigma, np.absolute(Ysigma - rf.predict(Xsigma)))
    ridge_sigma.fit(Xsigma, np.absolute(Ysigma - rf.predict(Xsigma)))

    model_list += [rf] * 6
    denom_list += [1, gb_sigma, rf_sigma, svr_sigma, lasso_sigma, ridge_sigma]

    modsel_selection_list = np.zeros((len(q_list), modelnum))
    for k in range(modelnum):
        reg = model_list[k]
        denom = denom_list[k]

        calib_scores_modsel = 1000 * (Ymodsel_calib > 0) - reg.predict(Xmodelsel_calib) / (1 if isinstance(denom, int) else denom.predict(Xmodelsel_calib))
        test_scores_modsel = -reg.predict(Xmodsel_test) / (1 if isinstance(denom, int) else denom.predict(Xmodsel_test))
        pvals = np.zeros(len(Xmodsel_test))
        for j in range(len(Xmodsel_test)):
            pvals[j] = (np.sum(calib_scores_modsel < test_scores_modsel[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores_modsel == test_scores_modsel[j]) + 1)) / (len(calib_scores_modsel) + 1)
        for qid, q in enumerate(q_list):
            try_sel = BH(pvals, q)
            modsel_selection_list[qid, k] = len(try_sel)

    for qid, q in enumerate(q_list):
        max_modsel_sel = max(modsel_selection_list[qid])
        theta_j = np.random.choice([a for a in range(modelnum) if modsel_selection_list[qid, a] == max_modsel_sel])

        reg = model_list[theta_j]
        denom = denom_list[theta_j]

        calib_scores = 1000 * (Ycalib_0 > 0) - reg.predict(Xcalib_0) / (1 if isinstance(denom, int) else denom.predict(Xcalib_0))
        test_scores = -reg.predict(Xtest) / (1 if isinstance(denom, int) else denom.predict(Xtest))

        baseline3_pvals = np.zeros(ntest)
        for j in range(ntest):
            baseline3_pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)

        baseline3_sel = BH(baseline3_pvals, q)
        Baseline3_FDP, Baseline3_power = eval(Ytest, baseline3_sel, 0, np.inf) 

        baseline3_df = pd.concat((baseline3_df, pd.DataFrame({
            'Baseline3_FDP': [Baseline3_FDP],
            'Baseline3_power': [Baseline3_power] 
        })))

    this_res = pd.concat([baseline1_df, baseline2_df, baseline3_df, naive_df, OptCS_df], axis=1)

    all_res = pd.concat((all_res, this_res))

if not os.path.exists('results'):
    os.makedirs('results')

all_res.to_csv(os.path.join('results', f"Jin, setting={setting}, modelnum={modelnum}, dim={dim}, sigma={sig}, ntest={ntest}, ncalib={ncalib}, ntrain={ntrain}, seed={seedgroup}.csv"))