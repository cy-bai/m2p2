# loss functions
import torch

def AlignmentLoss(out_mod, criterion, COSINE, MODS, device):
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