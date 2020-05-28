#!/usr/bin/env python
# coding: utf-8
import argparse
from torch import optim

import torch.nn as nn
import time
import pandas as pd

import utils
import model as m2p2
import train

parser = argparse.ArgumentParser()
parser.add_argument('--fd', required=False, default =0, type=int, help='fold id')
parser.add_argument('--mod', required=False, default ='avl', type=str,
                    help='modalities: a,v,l,or any combination of them')
parser.add_argument('--cos', required=False, default ='123', type=str,
                    help='alignment loss terms, '
                         '1 means cosine loss for a,v modalities, '
                         '2 means that for v,l, '
                         '3 means that for l,a')
parser.add_argument('--w_align', required=False, default =0.1, type=float, help='weight of alignment loss')
parser.add_argument('--nhead', required=False, default =4, type=int, help='# heads attention')
parser.add_argument('--nfeat', required=False, default =16, type=int, help='# feats embed')
parser.add_argument('--nlayers', required=False, default=1, type=int, help='# transformer encoder layers')
parser.add_argument('--dp', required=False, default=0.4, type=float, help='# dropout')
### boolean flags
parser.add_argument('--het_module', help='use reference models for weighted fusion', action='store_true')
parser.add_argument('--test_mode', help='test mode: loading a pre-trained model', action='store_true')
parser.add_argument('--verbose', help='print more information', action='store_true')

args = parser.parse_args()
FOLD = int(args.fd)
MODS = list(args.mod)
WITH_HET_MODULE = args.het_module # enable hetergeneity module or not
COSINE = [int(i) for i in list(args.cos)]
GAMMA = round(args.w_align, 2) # weight for alignment loss
print('het module:', WITH_HET_MODULE, ', alignment module loss weight:', GAMMA)
VERBOSE = args.verbose

MODEL_DIR = f'./new_trained_models/fold{FOLD}' # where to save your own trained model
PRETRAIN_MODEL_DIR = f'./pretrained_models/fold{FOLD}/' # where to load the provided pre-trained model
# load meta file
df = pd.read_csv(utils.META, index_col = 'seg_id')

nfeat = args.nfeat
DP= args.dp
# split the data to train, validation, test, according to fold FOLD
tra_loader, val_loader, tes_loader = utils.GetSplit(FOLD, df, MODS)

########## m2p2 model #########
# initialize models to output the latent embeddings for a,v,l
m2p2_model = {mod:m2p2.Emb(mod,nfeat, args.nhead, nfeat, dropout=DP,
                           nlayers=args.nlayers).to(utils.device) for mod in MODS}
# initialize models to predict persuasiveness given alignment embeddings, heterogeneity embeddings and debate meta-data
m2p2_model['pers'] = m2p2.Pers(nfeat, nmod = len(MODS), nhid = nfeat//2,
                               dropout=DP, is_pers_mlp = True).to(utils.device)
# initialize the shared mlp for alignemnt module
m2p2_model['align_mlp'] = m2p2.Align(nin = nfeat, nout = nfeat, dropout = DP).to(utils.device)
params = m2p2.trained_params(m2p2_model)
print('####### total trained #params', m2p2.count_trained_parameters(params))
m2p2_optim = optim.Adam(params, lr = utils.LR, weight_decay = utils.W_DCAY)
########## end of m2p2 model #########

########### ref model ##########
if not args.test_mode:
    # initialize unimodal reference models
    ref_model = {mod:m2p2.Pers(nfeat, nmod = 1, nhid = nfeat//2,
                           dropout = DP, is_ref_model = True).to(utils.device) for mod in MODS}
    ref_params = m2p2.trained_params(ref_model)
    # optimizer for the reference models
    ref_model_optim = optim.Adam(ref_params, lr=utils.LR, weight_decay=utils.W_DCAY)
########### end of ref model ##########

# alignment loss for each pair of modalities
cri_align = nn.CosineEmbeddingLoss(reduction = 'mean')
# persuasion MSE loss
cri_pers = nn.MSELoss()

if VERBOSE:
    for k,v in m2p2_model.items():
        print(v)
        print(m2p2.count_trained_parameters(v.parameters()))

min_loss = 1e5
# initialize concat weights: w_A, w_V, w_L
mod_weights = {mod: 1./len(MODS) for mod in MODS}

if not args.test_mode:
    ##### training process (master procedure in alg 1) ######
    for epoch in range(utils.N_EPOCHS):
        start_time = time.time()

        train_emb_loss, train_pers_loss = train.train_m2p2(m2p2_model, tra_loader, m2p2_optim, cri_align, cri_pers,
                                                           COSINE, mod_weights, GAMMA, False)

        val_emb_loss, val_pers_loss = train.train_m2p2(m2p2_model, val_loader, m2p2_optim, cri_align, cri_pers,
                                                           COSINE, mod_weights, GAMMA, True)

        val_loss = val_pers_loss

        # save the optimal main m2p2 when the validation loss is the minimal
        if val_loss < min_loss:
            min_loss = val_loss
            utils.SaveModel(m2p2_model, MODEL_DIR+'opt/', mod_weights)

        # train the reference models (slave procedure in alg 1)
        if WITH_HET_MODULE:
            for ref_epoch in range(utils.n_EPOCHS):
                _ = train.train_ref(m2p2_model, ref_model, cri_pers, tra_loader, ref_model_optim, False)
        # end of slave procedure

        # apply the trained reference models to get current concat weights
        tilde_mod_weights = train.train_ref(m2p2_model, ref_model, cri_pers, val_loader, ref_model_optim, True)
        # moving average by combing current concat weights with previous concat weights
        if WITH_HET_MODULE: utils.update_mod_weights(mod_weights, tilde_mod_weights)

        # gather information and print in verbose mode
        end_time = time.time()
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
        if epoch % 1 == 0 and VERBOSE:
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            if WITH_HET_MODULE:
                print('modality weights', mod_weights)
            print(f'\tTrain alignment loss:{train_emb_loss:.5f}\tTrain persuasion loss:{train_pers_loss:.5f}')
            print(f'\tVal alignment loss:{val_emb_loss:.5f}\tVal persuasion loss:{val_pers_loss:.5f}')
    ##### end of training process (master procedure in alg 1) #####

else:
    ##### load pre-trained model and test #####
    mod_weights = utils.LoadModelDict(m2p2_model, PRETRAIN_MODEL_DIR)
    tes_emb_loss, tes_pers_loss = train.train_m2p2(m2p2_model, tes_loader, m2p2_optim, cri_align, cri_pers,
                                                   COSINE, mod_weights, GAMMA, evaluate=True)
    print('MSE:',round(tes_pers_loss, 3))
    ##### end of testing #####
