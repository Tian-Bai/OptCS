import torch

import sys
from importlib import reload
import persist_to_disk as ptd
import os
ptd.config.set_project_path(os.path.abspath("."))
import tqdm
import pandas as pd
import numpy as np
import re
import utils
import seaborn as sns
import math
import random

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.linear_model import LinearRegression, Lasso, Ridge, QuantileRegressor
from quantile_forest import RandomForestQuantileRegressor
from sklearn.svm import SVR
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import NearestNeighbors
from utility import BH, eBH, eval

sns.set_theme()
sns.set_context("notebook")

import argparse
from _settings import GEN_PATHS

import matplotlib.pyplot as plt
import uq_bb
reload(uq_bb)

def get_model(mdl_idx):
    if mdl_idx in [0, 1, 2]:
        qt = 0.25 * (mdl_idx + 1)
        return RandomForestQuantileRegressor(max_depth=5, default_quantiles=qt, random_state=0), 1
    if mdl_idx in [3, 4, 5]:
        qt = 0.25 * (mdl_idx - 2)
        return QuantileRegressor(quantile=qt), 1
    if mdl_idx in [6, 9]:
        return RandomForestRegressor(max_depth=5), 1 if mdl_idx == 6 else Lasso(alpha=1)
    elif mdl_idx in [7, 10]:
        return SVR(kernel='rbf', gamma=0.1), 1 if mdl_idx == 7 else Lasso(alpha=1)
    elif mdl_idx in [8, 11]:
        return LinearRegression(), 1 if mdl_idx == 8 else Lasso(alpha=1)

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
    mdl, denom = get_model(mdl_idx)

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
            p = classifier.predict_proba(Xtrain)[:, 1]  # predicted 0 probability
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

        if mdl_idx < 9:
            mdl.fit(Xtrain, Ytrain)
        else:
            Xtrain, Xsigma, Ytrain, Ysigma = train_test_split(Xtrain, Ytrain, train_size=0.7) # split for error model
            mdl.fit(Xtrain, Ytrain)
            denom.fit(Xsigma, np.absolute(Ysigma - mdl.predict(Xsigma)))

        if Xtest is not None:
            pred_test[:, ii] = mdl.predict(Xtest) / (1 if isinstance(denom, int) else denom.predict(Xtest))
        X_loo = Xtotal[LOO_range[ii]].reshape((1, -1))
        pred_loo[ii] = mdl.predict(X_loo)[0] / (1 if isinstance(denom, int) else denom.predict(X_loo)[0]) # predict only the left out sample
        
    return pred_test, pred_loo

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cxr')
parser.add_argument('--model', type=str, default='trained')
parser.add_argument("--batch_num", default=50, type=int, required=True,
                        help="Number of batches")
parser.add_argument("--batch_size", default=10, type=int, required=True,
                    help="Number of questions in each batch")
parser.add_argument('--tune_size', type=int, default=100) # reserved for tuning features
args = parser.parse_args()

# python3 -u _fdr.py --data "triviaqa" --model "llama-2-13b-chat-hf" --N 2000 --split_pr 0.5 --split_pr_tune 0.2

model = args.model
data = args.data
batch_num = args.batch_num
batch_size = args.batch_size
tune_size = args.tune_size

modelnum = 12

num_gens = 10
acc_name = 'generations|rougeL|acc'

path = GEN_PATHS[data][model] 

summ_kwargs = {
    'u+ea': {'overall': True, 'use_conf': False},
    'u+ia': {'overall': False, 'use_conf': False},
    'c+ia': {'overall': False, 'use_conf': True},
}['c+ia']

uq_list = [
    # set 1
    'generations|numsets', 
    'lexical_sim',
    'semanticEntropy|unnorm', 
    # set 2
    'generations|spectral_eigv_clip|disagreement_w',
    'generations|eccentricity|disagreement_w',
    'generations|degree|disagreement_w',
    # set 3
    'generations|spectral_eigv_clip|agreement_w',
    'generations|eccentricity|agreement_w',
    'generations|degree|agreement_w',
    # set 4
    'generations|spectral_eigv_clip|jaccard',
    'generations|eccentricity|jaccard',
    'generations|degree|jaccard',
    'semanticEntropy|unnorm', 
    # cannot be used
    # 'self_prob',
]

print(f"tune: {tune_size}")

#################################################################
#################################################################

# reference kwargs
o = uq_bb.UQ_summ(path, batch_num=batch_num, batch_size=batch_size, clean=True, split='test', cal_size=tune_size, train_size=0, seed=0)

uq_kwargs_ref = summ_kwargs

if len(o.key) > 2:
    assert o.key[2] == 'test'
    self2 = o.__class__(o.path, o.batch_num, o.batch_size, o.key[1], 'val', o.key[3], o.key[3], o.key[5])
    self2.tunable_hyperparams = {k:v for k, v in o.tunable_hyperparams.items() if k in uq_list}
    tuned_hyperparams = self2._tune_params(num_gens=num_gens,
                                           metric=acc_name,
                                           overall=False, use_conf=True, curve='auarc')
    uq_kwargs_ref.update(tuned_hyperparams)
else:
    uq_kwargs_ref.update(o._get_default_params())

print(f'uq_kwargs_ref: {uq_kwargs_ref}')

# load confidence/uncertainty scores

uq_res = []
for uq_ in uq_list:
    _, individual_res = o.get_uq(name=uq_, num_gens=num_gens, **uq_kwargs_ref.get(uq_,{}))
    print(individual_res.to_numpy().shape)
    uq_res.append(individual_res.to_numpy())

print(uq_res.index)
all_ids = o.ids

uq_res = np.array(uq_res)
uq_res = np.swapaxes(uq_res,0,1)
print(f'shape of uq_res: {uq_res.shape}')

label = o.get_acc(acc_name)[1]

#################################################################
#################################################################

# to save the data
ids = [all_ids[_] for _ in np.arange(uq_res.shape[0])]

ids_gen = 0
Y = label.loc[ids,:].to_numpy(dtype=int)[:, ids_gen]
X = uq_res[np.arange(uq_res.shape[0]),:,ids_gen]

save_df = pd.DataFrame(X, columns=uq_list)
save_df['Response'] = Y

# Save the DataFrame to a CSV file
save_df.to_csv("saved_feature_and_label.csv", index=False)