from importlib import reload
import persist_to_disk as ptd
import os
ptd.config.set_project_path(os.path.abspath("."))
import pandas as pd
import numpy as np
import seaborn as sns

sns.set_theme()
sns.set_context("notebook")

import argparse
from _settings import GEN_PATHS

import matplotlib.pyplot as plt
import uq_bb
reload(uq_bb)

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