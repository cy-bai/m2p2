#!/usr/bin/env python
# coding: utf-8
import argparse
from torch import optim
from utils import *
from dataset import *
import model as myModel
from train import *

parser = argparse.ArgumentParser()
parser.add_argument('--fd', required=False, default =0, type=int, help='fold id')
parser.add_argument('--mod', required=False, default ='avl', type=str, help='modalities')
parser.add_argument('--cos', required=False, default ='123', type=str, help='cosine loss terms')
parser.add_argument('--lamda', required=False, default =0.01, type=float, help='weight of alignment loss')
parser.add_argument('--nhead', required=False, default =4, type=int, help='# heads attention')
parser.add_argument('--nfeat', required=False, default =16, type=int, help='latent embedding dimension')
parser.add_argument('--nlayers', required=False, default=1, type=int, help='# layers of the transformer encoder')
parser.add_argument('--dp', required=False, default=0.4, type=float, help='dropout')
### boolean flags
parser.add_argument('--wloss', help='use reference models for weighted fusion', action='store_true')
parser.add_argument('--test', help='test by loading a pre-trained model', action='store_true')
parser.add_argument('--verbose', help='print information', action='store_true')

args = parser.parse_args()
FOLD = int(args.fd)
MODS = list(args.mod)
LOSS_GUIDED = args.wloss
COSINE = [int(i) for i in list(args.cos)]
WEIGHT = round(args.lamda, 2)
print('loss guide', LOSS_GUIDED, WEIGHT)
VERBOSE = args.verbose

MODEL_DIR = f'./new_trained_models/fold{FOLD}' # where to save your own trained model
PRETRAIN_MODEL_DIR = f'./models/fold{FOLD}' # where to load the provided pre-trained model

# load meta file
df = pd.read_csv(META, index_col = 'seg_id')

nfeat = args.nfeat
DP= args.dp
# split the data to train, validation, test, according to fold FOLD
tra_data, tra_loader, val_loader, tes_loader = GetSplit(FOLD, df, MODS)
# initialize models to output latent embeddings
mod_model = {mod:myModel.Emb(mod,nfeat, args.nhead, nfeat, dropout=DP, nlayers=args.nlayers).to(device) for mod in MODS}
# initialize models to predict persuasiveness given latent embeddings
pers_nhid = nfeat//2
mod_model['pers'] = myModel.Pers(nfeat, nmod = len(MODS), nhid = pers_nhid, dropout=DP).to(device)
# initialize reference models
ref_model = {mod:myModel.Pers(nfeat, nmod = 1, nhid = pers_nhid, dropout = DP).to(device) for mod in MODS}

# alignment loss for each pair of modality
cri_align = nn.CosineEmbeddingLoss(reduction = 'mean')
# persuasion MSE loss
cri_pers = nn.MSELoss()

if VERBOSE:
    for k,v in mod_model.items():
        print(v)
        print(myModel.count_trained_parameters(v.parameters()))

params, pers_msk = myModel.trained_params(mod_model)
print('####### total trained #params', myModel.count_trained_parameters(params))
# optimizer for tha main model
main_optim = optim.Adam(params, lr = 1e-3, weight_decay = 1e-5)
ref_params,_ = myModel.trained_params(ref_model)
# optimizer for the reference models
ref_model_optim = optim.Adam(ref_params, lr = 1e-3 , weight_decay = 1e-5)

min_loss = 1e5
# initialize concat weights
concat_weights = {mod: 1./len(MODS) for mod in MODS}

if not args.test:
    ##### training process #####
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        # train the main model
        train_emb_loss, train_pers_loss = train(mod_model, tra_loader, main_optim, cri_align, cri_pers,
                                                COSINE, concat_weights, WEIGHT, False)
        val_emb_loss, val_pers_loss = train(mod_model, val_loader, main_optim, cri_align, cri_pers,
                                            COSINE, concat_weights, WEIGHT, True)

        # save the optimal main model when the validation loss is the minimal and the epoch exceeds half of NEPOCHS
        cur_loss = val_pers_loss
        if epoch > N_EPOCHS//2 and cur_loss < min_loss:
            min_loss = cur_loss
            SaveModel(mod_model, MODEL_DIR, concat_weights)

        # train the reference models
        if LOSS_GUIDED:
            for ref_epoch in range(N_REF_EPOCHS):
                _ = train_ref(mod_model, ref_model, cri_pers, tra_loader, ref_model_optim, False)
        # apply the trained reference models to get current concat weights
        cur_concat_weights = train_ref(mod_model, ref_model, cri_pers, val_loader, ref_model_optim, True)
        # combine current concat weights with previous concat weights
        if LOSS_GUIDED: update_concat_weights(concat_weights, cur_concat_weights)

        # gather information and print in verbose mode
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if epoch % 2 == 0 and VERBOSE:
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            if LOSS_GUIDED:
                print('concat weights', concat_weights)
            print(f'\tTrain alignment loss:{train_emb_loss:.5f}\tTrain persuasion loss:{train_pers_loss:.5f}')
            print(f'\tVal alignment loss:{val_emb_loss:.5f}\tVal persuasion loss:{val_pers_loss:.5f}')
    ##### end of training process #####
else:
    ##### load pre-trained model and test #####
    concat_weights = LoadModelDict(mod_model, PRETRAIN_MODEL_DIR)
    tes_emb_loss, tes_pers_loss = train(tes_loader, main_optim, criterion, concat_weights, evaluate = True)
    print(round(tes_pers_loss,3))
    ##### end of testing #####