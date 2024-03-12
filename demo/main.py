import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from time import time
import pandas as pd
import numpy as np
import tqdm
import random
from collections import defaultdict
import argparse

import torch
import torch.nn as nn
from torch import cuda
from torch.optim import Adam
from torch.utils.data import Dataset

from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss,DataLoader
from torchkge import KnowledgeGraph,DistMultModel,TransEModel,TransRModel
from torchkge.models.bilinear import HolEModel,ComplExModel

from utils import *

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='PertKG',
        usage='main.py [<args>] [-h | --help]'
    )
    parser.add_argument('--cause_file',default="./processed_data/target_inference_1/cause.txt")
    parser.add_argument('--process_file',default="./processed_data/knowledge_graph/process.txt")
    parser.add_argument('--effect_file', default="./processed_data/target_inference_1/effect.txt")
    parser.add_argument('--test_file',default="./processed_data/target_inference_1/test.txt")
    parser.add_argument('--seed', type = int, default=42)
    parser.add_argument('--h_dim', type = int, default=300)
    parser.add_argument('--margin', type = float, default=1.0)
    parser.add_argument('--lr', type = float, default=1e-4)
    parser.add_argument('--wd', type = float, default=1e-5)
    parser.add_argument('--n_neg', type = int, default=100)
    parser.add_argument('--batch_size', type = int, default=2048)
    parser.add_argument('--warm_up', type = int, default=10)
    parser.add_argument('--patients', type = int, default=5)
    parser.add_argument('--use_cuda', type = str, default='batch')
    parser.add_argument('--nepoch', type = int, default=100)
    parser.add_argument('--save_model', action='store_true', default=True)
    parser.add_argument('--save_model_path',default="./best_model/target_inference_1/")
    parser.add_argument('--load_processed_data', action='store_true', default=True)
    parser.add_argument('--processed_data_file',default="./processed_data/target_inference_1/")

    # +++++++++++++++++
    return parser.parse_args(args),parser.parse_args(args).__dict__


def five_fold_cv(args):
    set_seeds(args.seed)

    if args.save_model:
        if not os.path.exists(args.save_model_path):
                os.makedirs(args.save_model_path)

    # read cause, process, effect, test file
    cause, pertkg_wo_cause, test, ent2id, rel2id, pro2nc = read_files(args)

    # t_cand
    t_cand = [v for k,v in ent2id.items() if k.startswith(('Protein:','TF:','RBP:'))]
    print('total {} targets'.format(len(t_cand)))
    print('_'*50)
    
    # TODO:split cause according to compound
    five_fold_train, five_fold_valid = generate_five_fold_files(args, cause)

    results = []
    for i in range(5):
        # load data to consrtuct kg
        print('five_fold_cv_{}!!!!!!!!!!'.format(i))
            
        # TODO:改一下这里
        train = five_fold_train[i]
        valid = five_fold_valid[i]

        df = pd.concat([pertkg_wo_cause,train])
        df = df.sample(frac=1,random_state=42).reset_index(drop=True) # 打乱KG
        kg = KnowledgeGraph(df,ent2ix=ent2id,rel2ix=rel2id)
        print('chemical perturbation profiles-based knowledge graph was loaded!!!!!!!!!!')

        model = DistMultModel(args.h_dim, len(ent2id), len(rel2id))
        criterion = MarginLoss(args.margin)
        if cuda.is_available():
            cuda.empty_cache()
            model.cuda()
            criterion.cuda()
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        kgsampler = BernoulliNegativeSampler(kg,n_neg=args.n_neg)
        kgloader = DataLoader(kg, batch_size=args.batch_size, use_cuda=args.use_cuda)


        print('kge pretraining!!!!!!')
        best_hits100 = 0
        patients = 0
        for epoch in range(args.nepoch):
            # 训练kg
            running_loss = 0.0
            model.train()
            for batch in tqdm.tqdm(kgloader):
                h, t, r = batch[0], batch[1], batch[2]
                n_h, n_t = kgsampler.corrupt_batch(h, t, r)

                optimizer.zero_grad()

                # forward + backward + optimize
                pos, neg = model(h, t, r, n_h, n_t)
                loss = criterion(pos, neg)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            train_loss = running_loss
            print(
            'Epoch {} | train loss: {:.5f}'.format(epoch + 1,
                                                train_loss))  
            
            model.normalize_parameters()
            model.eval()
            with torch.no_grad():
                ent_emb,rel_emb = model.get_embeddings() # (n_ent, emb_dim)
                MR,MRR,Hit10,Hit30,Hit100 = unbiased_evaluator(valid,
                                                        ent2id,rel2id,
                                                        ent_emb,rel_emb,
                                                        pro2nc)

                print('Epoch {} | valid:'.format(epoch + 1))
                print('MR {} | MRR: {} | Hits@10:{} | Hits@30:{} | Hits@100: {}'.format(MR,
                                                                                MRR,
                                                                                Hit10,
                                                                                Hit30,
                                                                                Hit100))
            if epoch > args.warm_up:
                if Hit100 > best_hits100:  # 用MRR进行早停
                    best_hits100 = Hit100
                    patients = 0
                    best_result = [MR,MRR,Hit10,Hit30,Hit100]
                    if args.save_model:
                            torch.save(model.state_dict(), args.save_model_path + "pertkg{}.pt".format(i))

                else:
                    patients += 1

                if patients >= args.patients:
                    break
        
        # test
        print('testing now!!!')
        model.load_state_dict(torch.load(args.save_model_path + "pertkg{}.pt".format(i)))
        model.normalize_parameters()
        model.eval()
        with torch.no_grad():
            ent_emb,rel_emb = model.get_embeddings() # (n_ent, emb_dim)
            top_10, top_30, top_100, recall_10, recall_30, recall_100 = tester( test,
                                                            ent2id,rel2id,
                                                            ent_emb,rel_emb,
                                                            t_cand)
            print('Top-100:{} | Recall@100:{}'.format(top_100, recall_100))
        results.append([top_10, top_30, top_100, recall_10, recall_30, recall_100])

    df = pd.DataFrame(results, columns=['top_10', 'top_30', 'top_100', 'recall_10', 'recall_30', 'recall_100'])
    print(df.describe())


if __name__ == '__main__':
    s = time()

    args, args_dict = parse_args()
    print(args_dict)  # 加入打印参数

    five_fold_cv(args)

    e = time()
    print(f"Total running time: {round(e - s, 2)}s")




