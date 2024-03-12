import os

import datetime
import pandas as pd
import numpy as np
import tqdm
import random
from collections import defaultdict

import torch

def set_seeds(seed):  # seed只是固定了第n次的结果一样，所以在一次程序中第n次调用一个函数是不一样的
    "set random seeds"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

# 获得ent2id,rel2id
def get_dictionaries(df):
    """Build entities or relations dictionaries.
    Parameters
    ----------
    df: `pandas.DataFrame`
        Data frame containing three columns [from, to, rel].
    ent: bool
        if True then ent2ix is returned, if False then rel2ix is returned.
    Returns
    -------
    dict: dictionary
        Either ent2ix or rel2ix.
    """
    tmp1 = list(set(df['from'].unique()).union(set(df['to'].unique())))
    tmp2 = list(df['rel'].unique())
    return {ent: i for i, ent in enumerate(sorted(tmp1))},{rel: i for i, rel in enumerate(sorted(tmp2))}

# 生成负样本作为早停指标
def get_cpi(df):
    c2p_dict = defaultdict(set)
    p2c_dict = defaultdict(set)
    for i in range(len(df)):
        c = df.iloc[i]['from']
        p = df.iloc[i]['to']
        c2p_dict[c].add(p)
        p2c_dict[p].add(c)
    
    return c2p_dict,p2c_dict


def read_files(args):
    # read files
    cause = pd.read_csv(args.cause_file,sep='\t',names=['from','rel','to']).drop_duplicates()
    process = pd.read_csv(args.process_file,sep='\t',names=['from','rel','to']).drop_duplicates()
    effect = pd.read_csv(args.effect_file,sep='\t',names=['from','rel','to']).drop_duplicates()
    test = pd.read_csv(args.test_file,sep='\t',names=['from','rel','to']).drop_duplicates()

    # generate processed data
    if args.load_processed_data:
        ent2id = np.load(args.processed_data_file + 'ent2id.npy', allow_pickle=True).item()
        rel2id = np.load(args.processed_data_file + 'rel2id.npy', allow_pickle=True).item()
        pro2nc = np.load(args.processed_data_file + 'pro2nc.npy', allow_pickle=True).item()
    else:
        # 1.ent2id, rel2id
        pertkg = pd.concat([cause, process, effect])
        ent2id,rel2id = get_dictionaries(pertkg)
        np.save('./processed_data/ent2id.npy',ent2id)
        np.save('./processed_data/rel2id.npy',rel2id)
        # 2.pro2nc
        comp_cand = {k for k,v in ent2id.items() if k.startswith('CID:')}
        _,p2c_dict = get_cpi(cause)
        pro = set(cause['to'])
        pro2nc = {}
        for x in pro:
            pro2nc[x] = random.sample(comp_cand - p2c_dict[x],k=3000)
        np.save('./pro2nc.npy',pro2nc)


    pertkg_wo_cause = pd.concat([process,effect])

    return cause, pertkg_wo_cause, test, ent2id, rel2id, pro2nc

def generate_five_fold_files(args, cause):
    if args.load_processed_data:
        five_fold_train = []
        five_fold_valid = []
        for i in range(5):
            train = pd.read_csv(args.processed_data_file + 'train{}.txt'.format(i),sep='\t',names=['from','rel','to'])
            valid = pd.read_csv(args.processed_data_file + 'valid{}.txt'.format(i),sep='\t',names=['from','rel','to'])
            five_fold_train.append(train)
            five_fold_valid.append(valid)
    else:
        pass
    
    return five_fold_train, five_fold_valid


def count_mean_std(my_list):
    mean_value = np.mean(my_list)
    std_value = np.std(my_list)

    print("Mean:", mean_value)
    print("Standard Deviation:", std_value)

def count_n(df, n=10):
    comp = set(df['from'])
    c_recall = []
    c_top = []
    for x in comp:
        sdf = df[df['from'] == x]
        ssdf = sdf[sdf['rank']<= n]
        c_recall.append(len(ssdf)/len(sdf))
        if len(ssdf)> 0 :
            c_top.append(1)

    return np.mean(c_recall), len(c_top)/len(comp)

def unbiased_evaluator(df,
                ent2id, rel2id, 
                ent_emb, rel_emb,
                pro2nc):
    # used for unbiased early stopping

    h_ranks = []
    r_emb = rel_emb[rel2id['HAS_BINDING_TO']]
    for i in tqdm.tqdm(range(len(df))):
        c = df.iloc[i]['from']
        p = df.iloc[i]['to']
        
        #
        h_emb = ent_emb[ent2id[c]]
        t_emb = ent_emb[ent2id[p]]
        score = (h_emb * r_emb * t_emb).sum()
        
        # replace head
        h_cand = pro2nc[p]
        h_cand = [ent2id[x] for x in h_cand]
        h_cand_emb = ent_emb[h_cand]  # (3000,dim)
        h_score = (h_cand_emb * r_emb * t_emb).sum(dim=1)

        rank = int(sum(h_score >= score).cpu()) + 1
        h_ranks.append(rank)

    h_MR = np.mean(h_ranks)
    h_MRR = sum([1 / x for x in h_ranks]) / len(h_ranks)
    h_Hit10 = sum(1 for x in h_ranks if x<= 10) / len(h_ranks)
    h_Hit30 = sum(1 for x in h_ranks if x<= 30) / len(h_ranks)
    h_Hit100 = sum(1 for x in h_ranks if x<= 100) / len(h_ranks)


    return h_MR,h_MRR,h_Hit10,h_Hit30,h_Hit100

def tester(df,
           ent2id, rel2id, 
           ent_emb, rel_emb,
           t_cand):
    
    t_ranks = []
    r_emb = rel_emb[rel2id['HAS_BINDING_TO']]

    for i in tqdm.tqdm(range(len(df))):
        c = df.iloc[i]['from']
        p = df.iloc[i]['to']
        
        #
        h_emb = ent_emb[ent2id[c]]
        t_emb = ent_emb[ent2id[p]]
        score = (h_emb * r_emb * t_emb).sum()
        
        # replace head
        t_cand_emb = ent_emb[t_cand]  # (2000,dim)
        t_score = (h_emb * r_emb * t_cand_emb).sum(dim=1)

        rank = int(sum(t_score >= score).cpu())
        t_ranks.append(rank)
    
    t_ranks = [1 if x == 0 else x for x in t_ranks]  # ensure no zero rank 
    df['rank'] = t_ranks

    recall_10, top_10 = count_n(df,10)
    recall_30, top_30 = count_n(df,30)
    recall_100, top_100 = count_n(df,100)

    return top_10, top_30, top_100, recall_10, recall_30, recall_100