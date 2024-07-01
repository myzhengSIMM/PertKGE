import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.nn import functional as F, Parameter

from torch.utils.data import DataLoader
from torch.nn.init import xavier_uniform_

# This file is used for loading extra model not be included in TorchKGE package

def init_embedding(n_vectors, dim):
    """Create a torch.nn.Embedding object with `n_vectors` samples and `dim`
    dimensions. It is then initialized with Xavier uniform distribution.
    """
    entity_embeddings = Embedding(n_vectors, dim)
    xavier_uniform_(entity_embeddings.weight.data)

    return entity_embeddings


# model not implement by trochkge
class RotatEModel(nn.Module):
    def __init__(self, emb_dim, n_ent, n_rel):

        super(RotatEModel, self).__init__()


        self.re_ent_emb = init_embedding(n_ent, emb_dim)
        self.im_ent_emb = init_embedding(n_ent, emb_dim)
        self.rel_emb = init_embedding(n_rel, emb_dim)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(14) / 500]), 
            requires_grad=False
        )

    def forward(self, heads, tails, relations, negative_heads, negative_tails, negative_relations=None):
        """

        Parameters
        ----------
        heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's heads
        tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's tails.
        relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's relations.
        negative_heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled tails.ze)
        negative_relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled relations.

        Returns
        -------
        positive_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scoring function evaluated on true triples.
        negative_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scoring function evaluated on negatively sampled triples.

        """
        pos = self.scoring_function(heads, tails, relations)

        if negative_relations is None:
            negative_relations = relations

        if negative_heads.shape[0] > negative_relations.shape[0]:
            # in that case, several negative samples are sampled from each fact
            n_neg = int(negative_heads.shape[0] / negative_relations.shape[0])
            pos = pos.repeat(n_neg)
            neg = self.scoring_function(negative_heads,
                                        negative_tails,
                                        negative_relations.repeat(n_neg))
        else:
            neg = self.scoring_function(negative_heads,
                                        negative_tails,
                                        negative_relations)


        return pos, neg

    def scoring_function(self, h_idx, t_idx, r_idx):
        pi = 3.14159265358979323846

        embedding_vector = self.rel_emb(r_idx)
        phase_relation = embedding_vector/(self.embedding_range.item()/pi)
        self.re_rel_emb = torch.cos(phase_relation)
        self.im_rel_emb = torch.sin(phase_relation)
    
        re_score = self.re_ent_emb(h_idx) * self.re_ent_emb(t_idx) + self.im_rel_emb * self.im_ent_emb(t_idx)
        im_score = self.re_rel_emb * self.im_ent_emb(t_idx) - self.im_rel_emb * self.re_ent_emb(t_idx)
        re_score = re_score - self.re_ent_emb(h_idx)
        im_score = im_score - self.im_ent_emb(h_idx)

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = 12 - score.sum(dim = 1)

        return score

    def normalize_parameters(self):
        pass

    def get_embeddings(self):

        self.normalize_parameters()
        return self.re_ent_emb.weight.data, self.im_ent_emb.weight.data


class ConvEModel(nn.Module):
    def __init__(self, emb_dim, n_ent, n_rel,
                embedding_shape1,
                input_drop,hidden_drop,feat_drop,
                conv_bias,
                fc_dim,
                ):

        super(ConvEModel, self).__init__()


        self.ent_emb = init_embedding(n_ent, emb_dim)
        self.rel_emb = init_embedding(n_rel, emb_dim)
        self.emb_dim1 = embedding_shape1
        self.emb_dim2 = emb_dim // self.emb_dim1

        # multi dropout layer
        self.inp_drop = torch.nn.Dropout(input_drop)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(feat_drop)

        # conv layer
        self.conv1 = torch.nn.Conv2d(1, 5, (3, 3), 1, 0, bias=conv_bias)

        # bn layer
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(5)
        self.bn2 = torch.nn.BatchNorm1d(emb_dim)

        # fc layer
        self.fc = torch.nn.Linear(fc_dim, emb_dim)

        # bias
        self.ent_bias = Embedding(n_ent,1)
        self.ent_bias.weight.data.fill_(0)

    def forward(self, heads, tails, relations, negative_heads, negative_tails, negative_relations=None):
        """

        Parameters
        ----------
        heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's heads
        tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's tails.
        relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's relations.
        negative_heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled tails.ze)
        negative_relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled relations.

        Returns
        -------
        positive_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scoring function evaluated on true triples.
        negative_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scoring function evaluated on negatively sampled triples.

        """
        pos = self.scoring_function(heads, tails, relations)

        if negative_relations is None:
            negative_relations = relations

        if negative_heads.shape[0] > negative_relations.shape[0]:
            # in that case, several negative samples are sampled from each fact
            n_neg = int(negative_heads.shape[0] / negative_relations.shape[0])
            pos = pos.repeat(n_neg)
            neg = self.scoring_function(negative_heads,
                                        negative_tails,
                                        negative_relations.repeat(n_neg))
        else:
            neg = self.scoring_function(negative_heads,
                                        negative_tails,
                                        negative_relations)


        return pos, neg

    def scoring_function(self, h_idx, t_idx, r_idx):
        
        e1_embedded= self.ent_emb(h_idx).view(-1, 1, self.emb_dim1, self.emb_dim2)
        # print(e1_embedded.shape)
        rel_embedded = self.rel_emb(r_idx).view(-1, 1, self.emb_dim1, self.emb_dim2)
        e2_embedded = self.ent_emb(t_idx)
        bias = self.ent_bias(t_idx)
        # print(e2_embedded.shape)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        # print(stacked_inputs.shape)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)  # (batch, outcha*18*8)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        # print(x.shape)
        x = (x*e2_embedded).sum(dim=1).unsqueeze(dim=1)
        x = x+bias

        # readout
        score = torch.sigmoid(x.squeeze(dim=1))

        return score

    def normalize_parameters(self):
        pass

    def get_embeddings(self):
        self.normalize_parameters()

        return self.re_ent_emb.weight.data, self.im_ent_emb.weight.data


class MyModel(nn.Module):
    def __init__(self, emb_dim, n_ent, n_rel,
                ):

        super(MyModel, self).__init__()


        self.ent_emb = init_embedding(n_ent, emb_dim)
        self.rel_emb = init_embedding(n_rel, emb_dim)


    def forward(self, heads, tails, relations, negative_heads, negative_tails, negative_relations=None):
        """

        Parameters
        ----------
        heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's heads
        tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's tails.
        relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's relations.
        negative_heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled tails.ze)
        negative_relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled relations.

        Returns
        -------
        positive_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scoring function evaluated on true triples.
        negative_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scoring function evaluated on negatively sampled triples.

        """
        pos = self.scoring_function(heads, tails, relations)

        if negative_relations is None:
            negative_relations = relations

        if negative_heads.shape[0] > negative_relations.shape[0]:
            # in that case, several negative samples are sampled from each fact
            n_neg = int(negative_heads.shape[0] / negative_relations.shape[0])
            pos = pos.repeat(n_neg)
            neg = self.scoring_function(negative_heads,
                                        negative_tails,
                                        negative_relations.repeat(n_neg))
        else:
            neg = self.scoring_function(negative_heads,
                                        negative_tails,
                                        negative_relations)


        return pos, neg

    def scoring_function(self, h_idx, t_idx, r_idx):
        pass

    def normalize_parameters(self):
        pass

    def get_embeddings(self):
        self.normalize_parameters()

        return self.re_ent_emb.weight.data, self.im_ent_emb.weight.data
