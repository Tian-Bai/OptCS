import torch
from collections import OrderedDict
from transformers import BertConfig, BertModel, BertTokenizer
import torch
import torch.nn as nn
import pandas as pd

class CheXbert(nn.Module):
    def __init__(self, bert_path, chexbert_path, device, p=0.1):
        super(CheXbert, self).__init__()

        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        config = BertConfig().from_pretrained(bert_path)

        with torch.no_grad():

            self.bert = BertModel(config)
            self.dropout = nn.Dropout(p)

            hidden_size = self.bert.pooler.dense.in_features

            # Classes: present, absent, unknown, blank for 12 conditions + support devices
            self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])

            # Classes: yes, no for the 'no finding' observation
            self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

            # Load CheXbert checkpoint
            state_dict = torch.load(chexbert_path, map_location=device)['model_state_dict']

            new_state_dict = OrderedDict()
            # new_state_dict["bert.embeddings.position_ids"] = torch.arange(config.max_position_embeddings).expand((1, -1))
            for key, value in state_dict.items():
                if 'bert' in key:
                    new_key = key.replace('module.bert.', 'bert.')
                elif 'linear_heads' in key:
                    new_key = key.replace('module.linear_heads.', 'linear_heads.')
                new_state_dict[new_key] = value

            self.load_state_dict(new_state_dict)

        self.eval()

    def forward(self, reports):

        for i in range(len(reports)):
            reports[i] = reports[i].strip()
            reports[i] = reports[i].replace("\n", " ")
            reports[i] = reports[i].replace("\s+", " ")
            reports[i] = reports[i].replace("\s+(?=[\.,])", "")
            reports[i] = reports[i].strip()

        with torch.no_grad():

            tokenized = self.tokenizer(reports, padding='longest', max_length=512, return_tensors="pt", truncation=True)
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            last_hidden_state = self.bert(**tokenized)[0]

            cls = last_hidden_state[:, 0, :]
            cls = self.dropout(cls)

            predictions = []
            for i in range(14):
                predictions.append(self.linear_heads[i](cls).argmax(dim=1))

        return torch.stack(predictions, dim=1)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

chexbert_path = 'chexbert-checkpoint/chexbert.pth'
bert_path = "bert-base-uncased"

batch_size = 16

"""
0 = blank/not mentioned
1 = positive
2 = negative
3 = uncertain
"""

def chexbert_eval(y_hat, y, study_id):

    CONDITIONS = [
        'enlarged_cardiomediastinum',
        'cardiomegaly',
        'lung_opacity',
        'lung_lesion',
        'edema',
        'consolidation',
        'pneumonia',
        'atelectasis',
        'pneumothorax',
        'pleural_effusion',
        'pleural_other',
        'fracture',
        'support_devices',
        'no_finding',
    ]

    # with open(path_jsonl) as f:
    #     data = [json.loads(line) for line in f]

    model = CheXbert(
        bert_path=bert_path,
        chexbert_path=chexbert_path,
        device=device,
    ).to(device)

    table = {'chexbert_y_hat': [], 'chexbert_y': [], 'y_hat': [], 'y': [], 'study_id': []}
    # for batch in minibatch(data, batch_size):
    table['chexbert_y_hat'].extend([i + [j] for i, j in zip(model(list(y_hat)).tolist(), list(study_id))])
    table['chexbert_y'].extend([i + [j] for i, j in zip(model(list(y)).tolist(), list(study_id))])
    table['y_hat'].extend(y_hat)
    table['y'].extend(y)
    table['study_id'].extend(study_id)


    columns = CONDITIONS + ['study_id']
    df_y_hat = pd.DataFrame.from_records(table['chexbert_y_hat'], columns=columns)
    df_y = pd.DataFrame.from_records(table['chexbert_y'], columns=columns)

    # df_y_hat.to_csv(path_jsonl.replace('.jsonl', '_chexbert_y_hat.csv'))
    # df_y.to_csv(path_jsonl.replace('.jsonl', '_chexbert_y.csv'))

    df_y_hat = df_y_hat.drop(['study_id'], axis=1)
    df_y = df_y.drop(['study_id'], axis=1)

    df_y_hat = (df_y_hat == 1)
    df_y = (df_y == 1)

    tp = (df_y_hat == df_y).astype(float)

    fp = (df_y_hat == ~df_y).astype(float)
    fn = (~df_y_hat == df_y).astype(float)

    tp_cls = tp.sum()
    fp_cls = fp.sum()
    fn_cls = fn.sum()

    tp_eg = tp.sum(1)
    fp_eg = fp.sum(1)
    fn_eg = fn.sum(1)

    precision_class = (tp_cls / (tp_cls + fp_cls)).fillna(0)
    recall_class = (tp_cls / (tp_cls + fn_cls)).fillna(0)
    f1_class = (tp_cls / (tp_cls + 0.5 * (fp_cls + fn_cls))).fillna(0)

    scores = {
        'ce_precision_macro': precision_class.mean(),
        'ce_recall_macro': recall_class.mean(),
        'ce_f1_macro': f1_class.mean(),
        'ce_precision_micro': tp_cls.sum() / (tp_cls.sum() + fp_cls.sum()),
        'ce_recall_micro': tp_cls.sum() / (tp_cls.sum() + fn_cls.sum()),
        'ce_f1_micro': tp_cls.sum() / (tp_cls.sum() + 0.5 * (fp_cls.sum() + fn_cls.sum())),
        'ce_precision_example': (tp_eg / (tp_eg + fp_eg)).fillna(0).mean(),
        'ce_recall_example': (tp_eg / (tp_eg + fn_eg)).fillna(0).mean(),
        'ce_f1_example': (tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg))).fillna(0).mean(),
        'ce_num_examples': float(len(df_y_hat)),
    }

    class_scores_dict = {
       **{'ce_precision_' + k: v for k, v in precision_class.to_dict().items()},
       **{'ce_recall_' + k: v for k, v in recall_class.to_dict().items()},
       **{'ce_f1_' + k: v for k, v in f1_class.to_dict().items()},
    }
    # pd.DataFrame(class_scores_dict, index=['i',]).to_csv(path_jsonl.replace('.jsonl', '_chexbert_scores.csv'))

    tp = (df_y_hat == df_y).astype(float)
    tp_eg = tp.sum(1)

    return df_y, df_y_hat, f1_class, scores
    