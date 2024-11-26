# TODO: evaluate performances of different methods

from DeepPurpose import utils, CompoundPred
from tdc.utils import create_fold
from tdc.single_pred import ADME, Tox, HTS
import argparse
import pandas as pd
import pickle
import os
from utils import eval, BH, eBH
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import random

# similar parsing as in modelpred.py
adme_regression_dataset = ['caco2_wang', 'lipophilicity_astrazeneca', 'ppbr_az', 'vdss_lombardo', 'half_life_obach', 'clearance_microsome_az', 'clearance_hepatocyte_az']
tox_regression_dataset = ['ld50_zhu']

# classification adme datasets
adme_classification_dataset = ['hia_hou', 'bioavailability_ma', 'cyp3a4_substrate_carbonmangels', 'bbb_martins', 'cyp2d6_substrate_carbonmangels', 'cyp2c9_substrate_carbonmangels']
tox_classification_dataset = ['dili', 'herg']

parser = argparse.ArgumentParser('')

parser.add_argument('--data', type=str, default='caco2_wang', choices = adme_classification_dataset + adme_regression_dataset + tox_classification_dataset + tox_regression_dataset)
parser.add_argument('--itr', type=int, default=100)
parser.add_argument('--q', type=float, default=0.3)
parser.add_argument('--report_indiv', action="store_true", default=False)

args = parser.parse_args()
data_name = args.data
num_itr = args.itr # number of iterations (seeds)
report_indiv = args.report_indiv
q = args.q
task = 'admet'

# dataset-dependent selection threshold
# data_thresholds = {'caco2_wang': -5.1, 'lipophilicity_astrazeneca': 2.3, 'solubility_aqsoldb': -2.6, 'ppbr_az': 96, 
#                    'vdss_lombardo': 1.0, 'half_life_obach': 4.0, 'clearance_microsome_az': 12, 'clearance_hepatocyte_az': 20} # 50% good
data_thresholds = {'caco2_wang': -4.7, 'lipophilicity_astrazeneca': 2.9, 'solubility_aqsoldb': -1.5, 'ppbr_az': 98, 
                   'vdss_lombardo': 2.0, 'half_life_obach': 9.0, 'clearance_microsome_az': 33, 'clearance_hepatocyte_az': 50, 
                   'ld50_zhu': 2.9} # 30% good
deselect = False

if data_name in adme_regression_dataset + tox_regression_dataset:
    task_type = 'regression'
    threshold = data_thresholds[data_name]
else:
    task_type = 'classification'
    threshold = 0 # classification
    # for those datasets, there are too many good chemicals; therefore we 'deselect' and select the compounds with label 0
    if data_name in ['herg', 'bioavailability_ma', 'hia_hou']:
        deselect = True

# list the model/nonconformity choice pairs: [loss_type, encoding_method]
if task_type == 'classification':
    options = [('bce_loss', 'AttentiveFP'), ('bce_loss', 'Morgan'), ('bce_loss', 'CNN'), ('bce_loss', 'rdkit_2d'), 
               ('bce_loss', 'GCN'), ('bce_loss', 'NeuralFP'), ('bce_loss', 'GIN_AttrMasking'), ('bce_loss', 'GIN_ContextPred')]
else:
    # add different scores later
    options = [('mse_loss', 'AttentiveFP'), ('mse_loss', 'Morgan'), ('mse_loss', 'CNN'), ('mse_loss', 'rdkit_2d'), 
               ('mse_loss', 'GCN'), ('mse_loss', 'NeuralFP'), ('mse_loss', 'GIN_AttrMasking'), ('mse_loss', 'GIN_ContextPred'),
               ('quantile_regression_0.10', 'CNN'), ('quantile_regression_0.20', 'CNN'), ('quantile_regression_0.30', 'CNN'), ('quantile_regression_0.40', 'CNN'), ('quantile_regression_0.50', 'CNN'),
               ('quantile_regression_0.60', 'CNN'), ('quantile_regression_0.70', 'CNN'), ('quantile_regression_0.80', 'CNN'), ('quantile_regression_0.90', 'CNN')]

all_res = pd.DataFrame()

for i_itr in tqdm(range(1, num_itr+1)):
    random.seed(i_itr)

    ALL_DATA = {}

    loss, model = options[0]
    df = pd.read_csv(os.path.join('predicted_results', data_name, loss, model, data_name + '_random_seed1' + '_pred.csv'))
    df_calibtest = df[df['split'] == 'calibtest']
    ncalibtest = len(df_calibtest)
    calib_idx, test_idx = train_test_split(np.arange(ncalibtest), train_size=0.5)

    for opt in range(len(options)):
        loss, model = options[opt]

        # read the results
        df = pd.read_csv(os.path.join('predicted_results', data_name, loss, model, data_name + '_random_seed1' + '_pred.csv'))
        df_calibtest = df[df['split'] == 'calibtest']
        if loss.startswith('quantile'):
            # we don't need 'pred'; we will use 'lower' as the predicted conditional quantile
            df_calibtest['pred'] = df_calibtest['lower']

        # if the target is to select 0 instead of 1, simply apply lambda x: 1-x transform to all predicted results and labels
        if deselect:
            df_calibtest['Label'] = 1 - df_calibtest['Label']
            df_calibtest['pred'] = 1 - df_calibtest['pred']    
        
        # split calib and test
        df_calib = df_calibtest.iloc[calib_idx]
        df_test = df_calibtest.iloc[test_idx]

        ALL_DATA[(loss, model)] = (df_calib, df_test)

    ''' Baseline 1: randomly pick one model/nonconformity from the list of options '''
    option_idx = np.random.randint(0, len(options))
    loss, model = options[option_idx]

    df_calib, df_test = ALL_DATA[(loss, model)]
    ncalib, ntest = len(df_calib), len(df_test)

    Ycalib_true, Ycalib_pred = df_calib['Label'].to_numpy(), df_calib['pred'].to_numpy()
    Ytest_true, Ytest_pred = df_test['Label'].to_numpy(), df_test['pred'].to_numpy()

    # get the calibration scores, p-values, and evaluate
    calib_scores = 1000 * (Ycalib_true > threshold) + threshold * (Ycalib_true <= threshold) - Ycalib_pred
    test_scores = threshold - Ytest_pred

    pvals = np.zeros(ntest)
    for j in range(ntest):
        pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)
    baseline1_sel = BH(pvals, q)
    base1_FDP, base1_power = eval(Ytest_true, baseline1_sel, threshold, np.inf)

    ''' Baseline 2: Data split for model selection -  split calibration data '''
    # use the same splitting across options, get indices
    modsel_idxs, calib0_idxs = train_test_split(np.arange(ncalib), train_size=0.5)
    modsel_calib_idxs, modsel_test_idxs = train_test_split(modsel_idxs, train_size=0.5)

    modsel_selection_list = []

    for opt in range(len(options)):
        loss, model = options[opt]
        df_calib, df_test = ALL_DATA[(loss, model)]

        Ycalib_true, Ycalib_pred = df_calib['Label'].to_numpy(), df_calib['pred'].to_numpy()

        Ymodsel_calib_true, Ymodsel_test_true = Ycalib_true[modsel_calib_idxs], Ycalib_true[modsel_test_idxs]
        Ymodsel_calib_pred, Ymodsel_test_pred = Ycalib_pred[modsel_calib_idxs], Ycalib_pred[modsel_test_idxs]

        # use Ymodsel_calib as calibration and test performance on Ymodsel_test
        calib_scores = 1000 * (Ymodsel_calib_true > threshold) + threshold * (Ymodsel_calib_true <= threshold) - Ymodsel_calib_pred
        test_scores = threshold - Ymodsel_test_pred
        nmodsel_test = len(Ymodsel_test_pred)

        pvals = np.zeros(nmodsel_test)
        for j in range(nmodsel_test):
            pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)
        try_sel = BH(pvals, q)
        modsel_selection_list.append(len(try_sel))

    # select the best one
    max_modsel_sel = max(modsel_selection_list)
    theta_j = np.random.choice([a for a in range(len(options)) if modsel_selection_list[a] == max_modsel_sel])

    # now read the results and do evaluation
    loss, model = options[theta_j]
    df_calib, df_test = ALL_DATA[(loss, model)]

    Ycalib_true, Ycalib_pred = df_calib['Label'].to_numpy(), df_calib['pred'].to_numpy()
    Ytest_true, Ytest_pred = df_test['Label'].to_numpy(), df_test['pred'].to_numpy()

    Ycalib0_true = Ycalib_true[calib0_idxs]
    Ycalib0_pred = Ycalib_pred[calib0_idxs]

    # get the calibration scores, p-values, and evaluate
    calib_scores = 1000 * (Ycalib0_true > threshold) + threshold * (Ycalib0_true <= threshold) - Ycalib0_pred
    test_scores = threshold - Ytest_pred

    pvals = np.zeros(ntest)
    for j in range(ntest):
        pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)
    baseline2_sel = BH(pvals, q)
    base2_FDP, base2_power = eval(Ytest_true, baseline2_sel, threshold, np.inf)

    ''' Naive method: no correction '''
    naive_selection_list = []
    indiv_FDPs, indiv_powers = [], []
    
    for opt in range(len(options)):
        loss, model = options[opt]
        df_calib, df_test = ALL_DATA[(loss, model)]

        Ycalib_true, Ycalib_pred = df_calib['Label'].to_numpy(), df_calib['pred'].to_numpy()
        Ytest_true, Ytest_pred = df_test['Label'].to_numpy(), df_test['pred'].to_numpy()

        calib_scores = 1000 * (Ycalib_true > threshold) + threshold * (Ycalib_true <= threshold) - Ycalib_pred
        test_scores = threshold - Ytest_pred

        pvals = np.zeros(ntest)
        for j in range(ntest):
            pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)
        sel = BH(pvals, q)
        if report_indiv:
            indiv_FDP, indiv_power = eval(Ytest_true, sel, threshold, np.inf)
            indiv_FDPs.append(indiv_FDP)
            indiv_powers.append(indiv_power)
        naive_selection_list.append(sel)

    # pick one option that leads to the largest selection set (and break ties randomly)
    max_naive_sel = max(len(sel) for sel in naive_selection_list)
    naive_sel = naive_selection_list[np.random.choice([k for k in range(len(naive_selection_list)) if len(naive_selection_list[k]) == max_naive_sel])]
    naive_FDP, naive_power = eval(Ytest_true, naive_sel, threshold, np.inf)

    ''' Our purposed method OptCS '''
    pvals = []
    max_aux_sel_sizes = []

    # TEST_DIFF1, TEST_DIFF2 = 0, 0
    # AUX_SIZE, TEST_SIZE = 0, 0

    for j in range(ntest):
        aux_sel_sizes = []
        for opt in range(len(options)):
            loss, model = options[opt]
            df_calib, df_test = ALL_DATA[(loss, model)]

            Ycalib_true, Ycalib_pred = df_calib['Label'].to_numpy(), df_calib['pred'].to_numpy()
            Ytest_true, Ytest_pred = df_test['Label'].to_numpy(), df_test['pred'].to_numpy()

            calib_scores = 1000 * (Ycalib_true > threshold) + threshold * (Ycalib_true <= threshold) - Ycalib_pred
            test_scores = threshold - Ytest_pred
            
            aux_pvals = np.zeros(ntest)
            for l in range(ntest):
                if l != j:
                    aux_pvals[l] = (np.sum(calib_scores <= test_scores[l]) + (test_scores[j] <= test_scores[l])) / (len(calib_scores) + 1)
            aux_sel = BH(aux_pvals, q)
            aux_sel_sizes.append(len(aux_sel))

            # ONLY FOR TESTING: IS THE R_j's VERY DIFFERENT FROM R, THE SIMPLE BH SELECTION SET?
            # test_pvals = np.zeros(ntest)
            # for j in range(ntest):
            #     test_pvals[j] = (np.sum(calib_scores <= test_scores[j]) + 1) / (len(calib_scores) + 1)
            # test_sel = BH(test_pvals, q)

            # TEST_DIFF1 += len([x for x in aux_sel if x not in set(test_sel)])
            # TEST_DIFF2 += len([x for x in test_sel if x not in set(aux_sel)])
            # AUX_SIZE += len(aux_sel)
            # TEST_SIZE += len(test_sel)
        
        # determine the best model when considering X_{n+j}
        max_aux_sel = max(aux for aux in aux_sel_sizes)
        theta_j = np.random.choice([a for a in range(len(options)) if aux_sel_sizes[a] == max_aux_sel])
        max_aux_sel_sizes.append(max_aux_sel)

        # now use theta_j to get the p-value
        loss, model = options[theta_j]
        df_calib, df_test = ALL_DATA[(loss, model)]

        Ycalib_true, Ycalib_pred = df_calib['Label'].to_numpy(), df_calib['pred'].to_numpy()
        Ytest_true, Ytest_pred = df_test['Label'].to_numpy(), df_test['pred'].to_numpy()

        calib_scores = 1000 * (Ycalib_true > threshold) + threshold * (Ycalib_true <= threshold) - Ycalib_pred
        test_scores = threshold - Ytest_pred
        p_j = (1 + np.sum(calib_scores < test_scores[j])) / (len(calib_scores) + 1)
        pvals.append(p_j)

    # TEST
    # print(TEST_DIFF1 / ntest / len(options))
    # print(TEST_DIFF2 / ntest / len(options))
    # print(AUX_SIZE / ntest / len(options))
    # print(TEST_SIZE / ntest / len(options))

    max_aux_sel_sizes = np.array(max_aux_sel_sizes)
    pvals = np.array(pvals)

    # get the e-values
    evalues = (pvals <= q * max_aux_sel_sizes / ntest) / (q * max_aux_sel_sizes / ntest)
    evalues[evalues == 0] = 1e-9

    # deterministic pruning
    optcs_dtm_sel = eBH(evalues, q)
    optcs_dtm_FDP, optcs_dtm_power = eval(Ytest_true, optcs_dtm_sel, threshold, np.inf)

    # heterogeneous pruning (not proved)
    optcs_hete_sel = eBH(evalues / np.random.uniform(0,1,size=ntest), q)
    optcs_hete_FDP, optcs_hete_power = eval(Ytest_true, optcs_hete_sel, threshold, np.inf)

    # homogeneous pruning (not proved)
    optcs_homo_sel = eBH(evalues / np.random.uniform(0,1,size=1)[0], q)
    optcs_homo_FDP, optcs_homo_power = eval(Ytest_true, optcs_homo_sel, threshold, np.inf)

    # option of using beta as a multipler for q
    betas = [0.9]
    optcs_dtm_beta_FDPs, optcs_dtm_beta_powers = [], []
    optcs_hete_beta_FDPs, optcs_hete_beta_powers = [], []
    optcs_homo_beta_FDPs, optcs_homo_beta_powers = [], []
    for beta in betas:
        evalues = (pvals <= q * max_aux_sel_sizes * beta / ntest) / (q * max_aux_sel_sizes * beta / ntest)
        evalues[evalues == 0] = 1e-9

        # deterministic pruning
        optcs_dtm_sel = eBH(evalues, q)
        optcs_dtm_FDP, optcs_dtm_power = eval(Ytest_true, optcs_dtm_sel, threshold, np.inf)
        optcs_dtm_beta_FDPs.append(optcs_dtm_FDP)
        optcs_dtm_beta_powers.append(optcs_dtm_power)

        # heterogeneous pruning (not proved)
        optcs_hete_sel = eBH(evalues / np.random.uniform(0,1,size=ntest), q)
        optcs_hete_FDP, optcs_hete_power = eval(Ytest_true, optcs_hete_sel, threshold, np.inf)
        optcs_hete_beta_FDPs.append(optcs_hete_FDP)
        optcs_hete_beta_powers.append(optcs_hete_power)

        # homogeneous pruning (not proved)
        optcs_homo_sel = eBH(evalues / np.random.uniform(0,1,size=1)[0], q)
        optcs_homo_FDP, optcs_homo_power = eval(Ytest_true, optcs_homo_sel, threshold, np.inf)
        optcs_homo_beta_FDPs.append(optcs_homo_FDP)
        optcs_homo_beta_powers.append(optcs_homo_power)

    ''' Baseline 3: Data split for model selection - split training data '''
    # use the df2 to simulate this
    ALL_DATA = {}

    loss, model = options[0]
    df = pd.read_csv(os.path.join('predicted_results', data_name, loss, model, data_name + '_random_seed1' + '_pred.csv'))
    df_calibtest = df[df['split'] == 'calibtest']
    ncalibtest = len(df_calibtest)
    calib_idx, test_idx = train_test_split(np.arange(ncalibtest), train_size=5/7)

    for opt in range(len(options)):
        loss, model = options[opt]

        # read the results
        df = pd.read_csv(os.path.join('predicted_results', data_name, loss, model, data_name + '_random_seed1' + '_pred_2.csv')) # note the ".._2.csv"
        df_calibtest = df[df['split'] == 'calibtest']
        if loss.startswith('quantile'):
            # we don't need 'pred'; we will use 'lower' as the predicted conditional quantile
            df_calibtest['pred'] = df_calibtest['lower']

        # if the target is to select 0 instead of 1, simply apply lambda x: 1-x transform to all predicted results and labels
        if deselect:
            df_calibtest['Label'] = 1 - df_calibtest['Label']
            df_calibtest['pred'] = 1 - df_calibtest['pred']    
        
        # randomly split calib and test
        df_calib = df_calibtest.iloc[calib_idx]
        df_test = df_calibtest.iloc[test_idx]

        ALL_DATA[(loss, model)] = (df_calib, df_test)
        ncalib, ntest = len(df_calib), len(df_test)

    # use the same splitting across options, get indices
    modsel_idxs, calib0_idxs = train_test_split(np.arange(ncalib), train_size=3/5)
    modsel_calib_idxs, modsel_test_idxs = train_test_split(modsel_idxs, train_size=0.5)

    modsel_selection_list = []

    for opt in range(len(options)):
        loss, model = options[opt]
        df_calib, df_test = ALL_DATA[(loss, model)]

        Ycalib_true, Ycalib_pred = df_calib['Label'].to_numpy(), df_calib['pred'].to_numpy()

        Ymodsel_calib_true, Ymodsel_test_true = Ycalib_true[modsel_calib_idxs], Ycalib_true[modsel_test_idxs]
        Ymodsel_calib_pred, Ymodsel_test_pred = Ycalib_pred[modsel_calib_idxs], Ycalib_pred[modsel_test_idxs]

        # use Ymodsel_calib as calibration and test performance on Ymodsel_test
        calib_scores = 1000 * (Ymodsel_calib_true > threshold) + threshold * (Ymodsel_calib_true <= threshold) - Ymodsel_calib_pred
        test_scores = threshold - Ymodsel_test_pred
        nmodsel_test = len(Ymodsel_test_pred)

        pvals = np.zeros(nmodsel_test)
        for j in range(nmodsel_test):
            pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)
        try_sel = BH(pvals, q)
        modsel_selection_list.append(len(try_sel))

    # select the best one
    max_modsel_sel = max(modsel_selection_list)
    theta_j = np.random.choice([a for a in range(len(options)) if modsel_selection_list[a] == max_modsel_sel])

    # now read the results and do evaluation
    loss, model = options[theta_j]
    df_calib, df_test = ALL_DATA[(loss, model)]

    Ycalib_true, Ycalib_pred = df_calib['Label'].to_numpy(), df_calib['pred'].to_numpy()
    Ytest_true, Ytest_pred = df_test['Label'].to_numpy(), df_test['pred'].to_numpy()

    Ycalib0_true = Ycalib_true[calib0_idxs]
    Ycalib0_pred = Ycalib_pred[calib0_idxs]

    # get the calibration scores, p-values, and evaluate
    calib_scores = 1000 * (Ycalib0_true > threshold) + threshold * (Ycalib0_true <= threshold) - Ycalib0_pred
    test_scores = threshold - Ytest_pred

    pvals = np.zeros(ntest)
    for j in range(ntest):
        pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)
    baseline3_sel = BH(pvals, q)
    base3_FDP, base3_power = eval(Ytest_true, baseline3_sel, threshold, np.inf)

    df_res = pd.DataFrame({
        'Naive_FDP': [naive_FDP],
        'Naive_power': [naive_power],
        'Baseline1_FDP': [base1_FDP],
        'Baseline1_power': [base1_power],
        'Baseline2_FDP': [base2_FDP],
        'Baseline2_power': [base2_power],
        'Baseline3_FDP': [base3_FDP],
        'Baseline3_power': [base3_power],
        'OptCS_dtm_FDP': [optcs_dtm_FDP],
        'OptCS_dtm_power': [optcs_dtm_power],
        'OptCS_hete_FDP': [optcs_hete_FDP],
        'OptCS_hete_power': [optcs_hete_power],
        'OptCS_homo_FDP': [optcs_homo_FDP],
        'OptCS_homo_power': [optcs_homo_power],
        'q': [q],
        'task': [data_name],
        'tasktype': [task_type],
        'optionnum': [len(options)],
        'seed': [i_itr],
        'ntest': [ntest]
    })

    if report_indiv: 
        for k in range(1, len(options)+1):
            df_res[f'option{k}_FDP'] = indiv_FDPs[k-1]
            df_res[f'option{k}_power'] = indiv_powers[k-1]

    # add beta results
    for i, beta in enumerate(betas):
        df_res[f'OptCS_dtm_{beta:.2f}_FDP'] = optcs_dtm_beta_FDPs[i]
        df_res[f'OptCS_dtm_{beta:.2f}_power'] = optcs_dtm_beta_powers[i]

        df_res[f'OptCS_hete_{beta:.2f}_FDP'] = optcs_hete_beta_FDPs[i]
        df_res[f'OptCS_hete_{beta:.2f}_power'] = optcs_hete_beta_powers[i]

        df_res[f'OptCS_homo_{beta:.2f}_FDP'] = optcs_homo_beta_FDPs[i]
        df_res[f'OptCS_homo_{beta:.2f}_power'] = optcs_homo_beta_powers[i]

    all_res = pd.concat((all_res, df_res))

output_folder = os.path.join('evaluated_results', data_name)
output_path = os.path.join(output_folder, f'q={q}, optionnum={len(options)}.csv')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

all_res.to_csv(output_path, index = False)