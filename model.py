import math
import numpy as np
import copy
import random

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models
from torch.nn.modules.container import ModuleList
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# import libs.qp09utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# extract visual input embedding from VGG-features
class ImgFeat(nn.Module):
    def __init__(self, feat_dim ,dropout=0.1):
        super().__init__()
        ninp = 512
        self.fc = nn.Linear(ninp, feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        x = self.fc(src)
        x = self.bn(x)
        return F.relu(self.dropout(x))

# extract audio input embedding from COVAREP features
class AudioFeat(nn.Module):
    def __init__(self, ninp, feat_dim ,dropout = 0.1):
        super().__init__()
        self.fc = nn.Linear(ninp, feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        x = self.fc(src)
        x = self.bn(x)
        return F.relu(self.dropout(x))

# extract language input embedding from word embedding
class LangFeat(nn.Module):
    def __init__(self, ninp, feat_dim, dropout = 0.1):
        super().__init__()
        self.fc = nn.Linear(ninp, feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        x = self.fc(src)
        x = self.bn(x)
        return F.relu(self.dropout(x))

# positional encoder + transformer encoder
class TransformerEmb(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers=1, dropout=0.1):
        super(TransformerEmb, self).__init__()
        # self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        # 1-layer transformer encoder
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def forward(self, src, seq_msk):
        src *= math.sqrt(self.ninp)
        # positional encoder
        src = self.pos_encoder(src)
        # transformer encoder
        output = self.transformer_encoder(src, src_key_padding_mask=seq_msk)
        return output
# positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# The model from raw features to latent embeddings of one modality
class Emb(nn.Module):
    def __init__(self, mod, feat_dim=128, nhead=8, nhid=256, dropout = 0.1, nlayers=1):
        super(Emb, self).__init__()
        if mod == 'a':
            self.feat_exa = AudioFeat(73, feat_dim, dropout)
        elif mod =='v':
            self.feat_exa = ImgFeat(feat_dim, dropout)
        else:
            self.feat_exa = LangFeat(200, feat_dim, dropout)

        self.transformer_emb = TransformerEmb(feat_dim, nhead, nhid, nlayers, dropout)
    def forward(self, src, seq_msk):
        # N: batch size, S: sequence length
        N,S = src.size()[0], src.size()[1]
        feats = torch.stack(
            [self.feat_exa(src[i]) for i in range(N)],
            dim = 0).transpose(0,1)
        # feats: (S,N,D)
        seq = self.transformer_emb(feats, seq_msk)
        seq = F.relu(seq)
        # max_pool
        msked_tmp = seq * (~seq_msk.unsqueeze(-1).transpose(0,1)).float()
        out = torch.max(msked_tmp, dim = 0)[0]
        return out

# the persuasion prediction 3-layer MLP
class Pers(nn.Module):
    def __init__(self, feat_dim=128, nmod = 3, nhid = 16, nout = 1, dropout = 0.1):
        super(Pers, self).__init__()
        ninp = feat_dim * nmod
        self.fc1 = nn.Linear(ninp+2, 2*nhid)
        self.fc2 = nn.Linear(2*nhid, nhid)
        self.fc3 = nn.Linear(nhid, nout)
        self.dropout = nn.Dropout(dropout)
        self.sigm = nn.Sigmoid()

    def forward(self, in_mod, vote_st, dur):
        input = torch.cat([v for k, v in in_mod.items()], dim=1)
        input = torch.cat([input, vote_st.unsqueeze(1), dur.unsqueeze(1)], dim = 1)
        x = self.fc1(input)
        x = F.relu(self.dropout(x))
        x = self.fc2(x)
        x = F.relu(self.dropout(x))
        x = self.fc3(x)
        return self.sigm(x)


# MISC
def count_trained_parameters(params):
    return sum(p.numel() for p in params if p.requires_grad)
def trained_params(mod_model):
    params = []
    pers_msk = []
    for mod, model in mod_model.items():
        param = [p for p in model.parameters() if p.requires_grad]
        if mod == 'pers':
            pers_msk += [True] * len(param)
        else:
            pers_msk += [False] * len(param)
        params += param

    return params, np.array(pers_msk)