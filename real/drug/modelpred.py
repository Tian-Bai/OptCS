from DeepPurpose import utils, CompoundPred
from tdc.utils import create_fold
from tdc.single_pred import ADME, Tox, HTS
import argparse
import pandas as pd
import pickle
import random
import os

# regression adme datasets
adme_regression_dataset = ['caco2_wang', 'lipophilicity_astrazeneca', 'ppbr_az', 'vdss_lombardo',  'half_life_obach', 'clearance_microsome_az', 'clearance_hepatocyte_az']
tox_regression_dataset = ['ld50_zhu']

# classification adme datasets
adme_classification_dataset = ['hia_hou', 'bioavailability_ma', 'bbb_martins', 'cyp2d6_substrate_carbonmangels', 'cyp3a4_substrate_carbonmangels', 'cyp2c9_substrate_carbonmangels']
tox_classification_dataset = ['dili', 'herg']

parser = argparse.ArgumentParser('')

parser.add_argument('--model', type=str, default='DGL_AttentiveFP', choices = ['DGL_AttentiveFP', 'Morgan', 'CNN', 'rdkit_2d_normalized', 'DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred'])
parser.add_argument('--conformal_score', type=str, default='error', choices = ['error', 'quantile', 'bnn_loss', 'mcdropout'])
parser.add_argument('--split_fct', type=str, default='random', choices = ['random'])
parser.add_argument('--alpha', type=float, default=0.5)

parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--data', type=str, default='caco2_wang', choices = adme_classification_dataset + adme_regression_dataset + tox_classification_dataset + tox_regression_dataset)
parser.add_argument('--pretrain', action="store_true", default=False)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--get_hid_emb', action="store_true", default=False)

args = parser.parse_args()
split_fct = args.split_fct
device = args.device
data_name = args.data
drug_encoding = args.model
seed = args.seed

random.seed(seed)

task = 'admet'
if data_name in adme_regression_dataset + tox_regression_dataset:
    task_type = 'regression'
else:
    task_type = 'classification'

if task_type == 'regression':
    if args.conformal_score == 'error':
        score_type = 'mse_loss'
    elif args.conformal_score == 'quantile':
        score_type = f'quantile_regression_{args.alpha:.2f}'
    elif args.conformal_score == 'bnn_loss':
        score_type = 'bnn_loss'
else:
    if args.conformal_score == 'error':
        score_type = 'bce_loss'
    else:
        score_type = args.conformal_score
        
if data_name in adme_regression_dataset + adme_classification_dataset:
    df = ADME(name=data_name).get_data()
elif data_name in tox_classification_dataset + tox_regression_dataset:
    df = Tox(name=data_name).get_data()
batch_size = 128

# filter the molecules
from rdkit import Chem
from tqdm import tqdm
tqdm.pandas()
def check_mol(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    else:
        return True
df['molecule_check'] = df.Drug.progress_apply(lambda x: check_mol(x))
df = df[df.molecule_check].reset_index(drop = True)
df2 = df.copy(deep=True)

# only use random split
# df: this is the version with more training data (for all method except baseline 3 which split training data). (train, valid, calibtest) = (50%, 10%, (20+20)%)
# df2: use less training data (for baseline 3), (train, valid, calibtest) = (25%, 5%, (30+20+20)%)
# the split for calib + test will be done in evaluate.py

if args.split_fct == 'random':
    df_split = create_fold(df, seed, [0.5, 0.1, 0.4])
    df_train = df_split['train']
    df_val = df_split['valid']
    df_calibtest = df_split['test'] # calib + test

    df2_split = create_fold(df2, seed, [0.25, 0.05, 0.7])
    df2_train = df2_split['train']
    df2_val = df2_split['valid']
    df2_calibtest = df2_split['test'] # calib + test
    
# process the data according to given encoding
def dp_data_process(df_train, df_val, df_calibtest, drug_encoding):

    train = utils.data_process(X_drug = df_train.Drug.values, y = df_train.Y.values, 
                            drug_encoding = drug_encoding,
                            split_method='no_split')
    val = utils.data_process(X_drug = df_val.Drug.values, y = df_val.Y.values, 
                            drug_encoding = drug_encoding,
                            split_method='no_split')
    calibtest = utils.data_process(X_drug = df_calibtest.Drug.values, y = df_calibtest.Y.values, 
                            drug_encoding = drug_encoding,
                            split_method='no_split')
    return train, val, calibtest  

# processed df's
train, val, calibtest = dp_data_process(df_train, df_val, df_calibtest, drug_encoding)
train2, val2, calibtest2 = dp_data_process(df2_train, df2_val, df2_calibtest, drug_encoding)

config = utils.generate_config(drug_encoding = drug_encoding, 
                               train_epoch = args.epoch, 
                               batch_size = batch_size)
config['device'] = device

if args.conformal_score == 'quantile':
    if args.data in adme_regression_dataset + tox_regression_dataset:
        config['quantile_regression'] = True
        config['quantile_regression_alpha'] = args.alpha
        # alpha could be larger than 0.5. We should always use the 'lower' as the predicted quantile as it corresponds to alpha.
    else:
        raise ValueError("Quantile regression does not apply for classification!")
elif args.conformal_score == 'bnn_loss':
    config['normal_loss'] = True
    
if args.conformal_score == 'quantile':
    save_model_path = os.path.join('model', data_name + '_' + drug_encoding + '_' + split_fct + '_seed' + str(seed) + '_quantile_regression')
    save_model_path2 = os.path.join('model', data_name + '_' + drug_encoding + '_' + split_fct + '_seed' + str(seed) + '_quantile_regression_2')
elif args.conformal_score == 'bnn_loss':
    save_model_path = os.path.join('model', data_name + '_' + drug_encoding + '_' + split_fct + '_seed' + str(seed) + '_bnn_loss')
    save_model_path2 = os.path.join('model', data_name + '_' + drug_encoding + '_' + split_fct + '_seed' + str(seed) + '_bnn_loss_2')
else:
    save_model_path = os.path.join('model', data_name + '_' + drug_encoding + '_' + split_fct + '_seed' + str(seed))
    save_model_path2 = os.path.join('model', data_name + '_' + drug_encoding + '_' + split_fct + '_seed' + str(seed) + '_2')
    
if args.pretrain:
    model = CompoundPred.model_pretrained(save_model_path)
else:    
    config['gnn_hid_dim_drug'] = 512
    model = CompoundPred.model_initialize(**config)
    model.train(train, val)    
    model.save_model(save_model_path)

    model2 = CompoundPred.model_initialize(**config)
    model2.train(train2, val2)
    model2.save_model(save_model_path2)

if args.conformal_score == 'quantile':
    
    calibtest['pred'], calibtest['lower'], calibtest['upper'] = model.predict(calibtest)
    train['pred'], train['lower'], train['upper'] = model.predict(train)
    val['pred'], val['lower'], val['upper'] = model.predict(val)

    calibtest2['pred'], calibtest2['lower'], calibtest2['upper'] = model.predict(calibtest2)
    train2['pred'], train2['lower'], train2['upper'] = model.predict(train2)
    val2['pred'], val2['lower'], val2['upper'] = model.predict(val2)
elif args.conformal_score == 'bnn_loss':
    calibtest['pred'], calibtest['logvar'] = model.predict(calibtest)
    train['pred'], train['logvar'] = model.predict(train)
    val['pred'], val['logvar'] = model.predict(val)

    calibtest2['pred'], calibtest2['logvar'] = model.predict(calibtest2)
    train2['pred'], train2['logvar'] = model.predict(train2)
    val2['pred'], val2['logvar'] = model.predict(val2)
else:
    calibtest['pred'] = model.predict(calibtest)
    train['pred'] = model.predict(train)
    val['pred'] = model.predict(val)

    calibtest2['pred'] = model.predict(calibtest2)
    train2['pred'] = model.predict(train2)
    val2['pred'] = model.predict(val2)

train['split'] = 'train'
calibtest['split'] = 'calibtest'
val['split'] = 'val'

train2['split'] = 'train'
calibtest2['split'] = 'calibtest'
val2['split'] = 'val'

import pandas as pd
df = pd.concat([train, val, calibtest])
df2 = pd.concat([train2, val2, calibtest2])

if args.conformal_score == 'quantile':
    df = df[['SMILES', 'Label', 'split', 'pred', 'lower', 'upper']]
    df2 = df2[['SMILES', 'Label', 'split', 'pred', 'lower', 'upper']]
elif args.conformal_score == 'bnn_loss':
    df = df[['SMILES', 'Label', 'split', 'pred', 'logvar']]
    df2 = df2[['SMILES', 'Label', 'split', 'pred', 'logvar']]
else:
    df = df[['SMILES', 'Label', 'split', 'pred']]
    df2 = df2[['SMILES', 'Label', 'split', 'pred']]
    
if drug_encoding[:3] == 'DGL':
    drug_encoding = drug_encoding[4:]
    
if drug_encoding == 'rdkit_2d_normalized':
    drug_encoding = 'rdkit_2d'

save_folder = os.path.join('predicted_results', data_name, score_type, drug_encoding)
save_path = os.path.join(save_folder, data_name + '_' + split_fct + '_seed' + str(seed) + '_pred.csv')
save_path2 = os.path.join(save_folder, data_name + '_' + split_fct + '_seed' + str(seed) + '_pred_2.csv')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

df.to_csv(save_path, index = False)
df2.to_csv(save_path2, index = False)

if args.get_hid_emb:
    df_ = pd.concat([train, val, calibtest])
    df_2 = pd.concat([train2, val2, calibtest2])
        
    hid = model.get_hidden_emb(df_)
    hid2 = model2.get_hidden_emb(df_2)
    print(hid.shape)

    save_path = os.path.join(save_folder, data_name + '_' + split_fct + '_seed' + str(seed) + '_hid.pkl')
    save_path2 = os.path.join(save_folder, data_name + '_' + split_fct + '_seed' + str(seed) + '_hid_2.pkl')
    
    with open(save_path, 'wb') as f:
        pickle.dump(hid, f)
    with open(save_path2, 'wb') as f:
        pickle.dump(hid2, f)