import os,sys,glob,json,time
import numpy as np
import scipy.io as sio
import pandas as pd
from scipy.spatial.distance import cosine as spcosine
import pickle
from collections import OrderedDict
from random import shuffle

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from dataset import *
from torch.utils.data import Sampler, Dataset, DataLoader

FOLDS_DIR = './folds_split/'

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def mkDir(fnm):
    if not os.path.exists(fnm):
        os.makedirs(fnm)
    return

def load_segs(fold, fnm, key):
    full_fnm = f'{FOLDS_DIR}/{key}/{fold}/{fnm}.data'
    segs = pickle.load(open(full_fnm,'rb'))
    return segs

def GetSplit(fold, df, MODS):
    tra_seg_file = '5_2'  # key for loading the training segments
    eval_seg_file = '5'  # key for loading the validation and testing segments
    tra_data = qpDataset(MODS, df, load_segs(fold, tra_seg_file, 'train'))
    val_data = qpDataset(MODS, df, load_segs(fold, eval_seg_file, 'val'))
    tes_data = qpDataset(MODS, df, load_segs(fold, eval_seg_file, 'tes'))

    tra_loader = DataLoader(tra_data, batch_size=BATCH, shuffle=True, num_workers=N_WORKERS, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=BATCH, shuffle=False, num_workers=N_WORKERS, drop_last=False)
    tes_loader = DataLoader(tes_data, batch_size=BATCH, shuffle=False, num_workers=N_WORKERS, drop_last=False)

    return tra_data, tra_loader, val_loader, tes_loader

def ModelMode(model_dict, evaluate = False):
    if evaluate:
        for k,model in model_dict.items():
            model.eval()
    else:
        for k,model in model_dict.items():
            model.train()

def SaveModel(mod_model, MODEL_DIR, concat_weights):
    mkDir(MODEL_DIR)
    for k,v in mod_model.items():
        torch.save(v, f'{MODEL_DIR}/{k}')
    np.savetxt(f'{MODEL_DIR}/concat_weights.txt', np.array(list(concat_weights.values())))

def LoadModelDict(mod_model, MODEL_DIR):
    for k in ['a','v','l','pers']:
        fnm = f'{MODEL_DIR}/{k}'
        if os.path.isfile(fnm):
            # print(fnm)
            save_dict = torch.load(fnm)
            for name, param in save_dict.items():
                if name in mod_model[k].state_dict():
                    mod_model[k].state_dict()[name].copy_(param)
            # mod_model[k].load_state_dict(torch.load(fnm))
    concat_weights = np.loadtxt(f'{MODEL_DIR}/concat_weights.txt')
    return {'a':concat_weights[0],'v':concat_weights[1],'l':concat_weights[2]}

def updateLR(optim, scale=5):
    # decrease lr by 5 times
    for g in optim.param_groups:
        g['lr'] /= scale

def update_concat_weights(weights, cur_weights):
    for k,v in weights.items():
        weights[k] = UPD_WEIGHT * weights[k] + (1-UPD_WEIGHT) * cur_weights[k]

def AlignmentLoss(out_mod, criterion, COSINE, MODS):
    N = out_mod[MODS[0]].shape[0]
    y = torch.ones(N).to(device)
    l1,l2,l3 = 0,0,0
    if 1 in COSINE and 'a' in MODS and 'v' in MODS:
        l1 = criterion(out_mod['a'], out_mod['v'], y)
    if 2 in COSINE and 'v' in MODS and 'l' in MODS:
        l2 = criterion(out_mod['v'], out_mod['l'], y)
    if 3 in COSINE and 'a' in MODS and 'l' in MODS:
        l3 = criterion(out_mod['a'], out_mod['l'], y)
    return l1 + l2 + l3

def PersLoss(y_pred, y_true, criterion):
    return criterion(y_pred[:,0], y_true)

# dataset index file
META = './qps_index.csv'
# employed hyperparameters & constants
UPD_WEIGHT = 0.5
BATCH = 32
N_WORKERS = 4
MAX_DUR = 482 # we devided all speaking length by this max value to normalize
N_EPOCHS = 300
N_REF_EPOCHS = 2
BETA = 50 # weight in the softmax function for concatenation weights

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(3)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic = True
print(device)