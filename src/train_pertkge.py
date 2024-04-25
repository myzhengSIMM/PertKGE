import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument('--cause_file',default="../processed_data/ddr1/cause.txt")
    parser.add_argument('--process_file',default="../processed_data/knowledge_graph/process.txt")
    parser.add_argument('--effect_file', default="../processed_data/ddr1/effect.txt")
    parser.add_argument('--test_file',default="../processed_data/ddr1/test.txt")
    parser.add_argument('--seed', type = int, default=43)
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
    parser.add_argument('--save_model_path',default="../best_model/ddr1/")
    parser.add_argument('--load_processed_data', action='store_true', default=False)
    parser.add_argument('--processed_data_file',default="../processed_data/ddr1/")
    parser.add_argument('--mode', default="no_test", help = 'choose reproduce if user want to report testing results')  # test or not
    parser.add_argument('--task', default="target_inference", help="choose from ['target_inference', 'virtual_screening', 'unbiased_test']")
    parser.add_argument('--run_name', default="ddr1_b", help="Name of the running.")

    # +++++++++++++++++
    return parser.parse_args(args),parser.parse_args(args).__dict__


def five_fold_cv(args):
    # read cause, process, effect, test file
    cause, pertkg_wo_cause, test, ent2id, rel2id, pro2nc, h_cand, t_cand = read_files(args)
    
    # generate train\valid
    five_fold_train, five_fold_valid = generate_five_fold_files(args, cause)

    results = []
    for i in range(5):
        # load data to consrtuct kg
        print('split_{}!!!'.format(i))
        
        # logger
        train_logger = SummaryWriter('../outlog/{}/split_{}'.format(args.run_name,i))

        # loading train and valid df
        train = five_fold_train[i]
        valid = five_fold_valid[i]

        print('construct chemical perturbation profiles-based knowledge graph!!!')
        s1 = time()
        df = pd.concat([pertkg_wo_cause,train])
        df = df.sample(frac=1,random_state=42).reset_index(drop=True) # 打乱KG
        kg = KnowledgeGraph(df,ent2ix=ent2id,rel2ix=rel2id)
        e1 = time()
        print(f"Total constructing time: {round(e1 - s1, 2)}s")
        print()

        print('split_{} traing now!!!'.format(i))
        model = DistMultModel(args.h_dim, len(ent2id), len(rel2id))
        criterion = MarginLoss(args.margin)
        if cuda.is_available():
            cuda.empty_cache()
            model.cuda()
            criterion.cuda()
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        kgsampler = BernoulliNegativeSampler(kg,n_neg=args.n_neg)
        kgloader = DataLoader(kg, batch_size=args.batch_size, use_cuda=args.use_cuda)

        best_hits100 = 0
        patients = 0
        for epoch in range(args.nepoch):
            # train kg
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

                train_logger.add_scalar("Hits@100", Hit100, epoch+1)

                print('Epoch {} | valid:'.format(epoch + 1))
                print('MR {} | MRR: {} | Hits@10:{} | Hits@30:{} | Hits@100: {}'.format(MR,
                                                                                MRR,
                                                                                Hit10,
                                                                                Hit30,
                                                                                Hit100))
            if epoch > args.warm_up:
                if Hit100 > best_hits100:  # Hits@100 is used as metric for early stopping
                    best_hits100 = Hit100
                    patients = 0
                    if args.save_model:
                            torch.save(model.state_dict(), args.save_model_path + "pertkg{}.pt".format(i))

                else:
                    patients += 1

                if patients >= args.patients:
                    break

        train_logger.flush()            

        if args.mode == 'reproduce':
            # report test metrics according to task
            print('split_{} testing now!!!'.format(i))
            model.load_state_dict(torch.load(args.save_model_path + "pertkg{}.pt".format(i)))
            model.normalize_parameters()
            model.eval()
            with torch.no_grad():
                ent_emb,rel_emb = model.get_embeddings() # (n_ent, emb_dim)
                metrics = tester(test,
                                ent2id,rel2id,
                                ent_emb,rel_emb,
                                h_cand,t_cand,
                                args.task)
                results.append(metrics)
            print('_'*50)
    
    if args.mode == 'reproduce':
        # report mean±std
        print('report mean±std testing results using 5 trained model!!!')
        if args.task == 'target_inference':
            df = pd.DataFrame(results, columns=['Top-10', 'Recall@10', 'Top-30', 'Recall@30', 'Top-100', 'Recall@100'])
            print(df.describe())
        
        elif args.task == 'virtual_screening':
            print('because ef is varied across different target, so we count metrics like unbiased_test here. using inference file for ef metrics.')
            df = pd.DataFrame(results, columns=['Hits@10', 'Hits@30', 'Hits@100'])
            print(df.describe())

        elif args.task == 'unbiased_test':
            df = pd.DataFrame(results, columns=['Hits@10', 'Hits@30', 'Hits@100'])
            print(df.describe())

        else:
            print('no testing metrics because task is not defined, plz run inference.ipynb to reload best_model for testing !!!')
        print('_'*50)

if __name__ == '__main__':
    s = time()

    print('print args_dict!!!')
    args, args_dict = parse_args()
    print(args_dict) 
    print('_'*50)

    set_seeds(args.seed)
    # save model
    if args.save_model:
        if not os.path.exists(args.save_model_path):
                os.makedirs(args.save_model_path)

    print('traing and testing using five-fold cross validation stategy!!!')
    print('_'*50)
    five_fold_cv(args)

    e = time()
    print(f"Total running time: {round(e - s, 2)}s")




