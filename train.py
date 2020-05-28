# functions for training
from scipy.special import softmax as spsoftmax
from utils import *

# train/validate the reference models
# if evaluate is true, execute reference
# otherwise, execute training
def train_ref(mod_model, ref_model, cri_pers, iterator, optim, evaluate = False):
    MODS = [key for key in mod_model.keys() if key != 'pers'] # available modalities
    ModelMode(ref_model, evaluate)
    # fix the main model
    ModelMode(mod_model, evaluate = True) # fix the original model
    pers_loss = {mod:0 for mod in MODS}

    for i_batch, sample_batched in enumerate(iterator):
        if not evaluate: optim.zero_grad()
        # get compact embeddings
        out = {}
        for mod in MODS:
            with torch.no_grad():
                out[mod] = mod_model[mod](sample_batched[f'{mod}_data'].to(device),
                                      sample_batched[f'{mod}_msk'].to(device))
        #st_vote = sample_batched['ed_vote'] - sample_batched['change']
        #dur = sample_batched['dur'].float().to(device) / MAX_DUR

        y_true = sample_batched['ed_vote'].float().to(device)

        # get predictions
        batch_loss = {}
        for mod in MODS:
            y_pred = ref_model[mod]({mod:out[mod]})
            # y_pred = ref_model[mod]({mod:out[mod]}, st_vote.float().to(device), dur)
            batch_loss[mod] = PersLoss(y_pred, y_true, cri_pers)
            pers_loss[mod] += batch_loss[mod].item()
        if not evaluate:
            sum(list(batch_loss.values())).backward()
            optim.step()

    loss_weights = spsoftmax(-BETA * np.array(list(pers_loss.values())) / (i_batch+1))
    return dict(zip(list(pers_loss.keys()), loss_weights))

def train(mod_model, iterator, opt, cri_align, cri_pers, COSINE, emb_wgts, alignment_weight, evaluate = False):
    MODS = [key for key in mod_model.keys() if key != 'pers']  # available modalities
    train_emb = (alignment_weight > 0)
    # set model mode
    ModelMode(mod_model, evaluate)
    # if evaluate:
    #     train_emb = False
    epoch_emb_loss = 0
    epoch_pers_loss = 0

    for i_batch, sample_batched in enumerate(iterator):
        if not evaluate: opt.zero_grad()
        # get embeddings
        out = {}
        for mod in MODS:
            if not evaluate:
                out[mod] = mod_model[mod](sample_batched[f'{mod}_data'].to(device),
                                          sample_batched[f'{mod}_msk'].to(device))
            else:
                with torch.no_grad():
                    out[mod] = mod_model[mod](sample_batched[f'{mod}_data'].to(device),
                                      sample_batched[f'{mod}_msk'].to(device))

        for mod in MODS:
            out[mod] *= torch.tensor(emb_wgts[mod])

        dur = sample_batched['dur'].float().to(device) / MAX_DUR
        st_vote = sample_batched['ed_vote'] - sample_batched['change']
        y_true = sample_batched['ed_vote'].float().to(device)

        # get predictions
        if not evaluate: y_pred = mod_model['pers'](out, st_vote.float().to(device), dur)
        else:
            with torch.no_grad():
                y_pred = mod_model['pers'](out, st_vote.float().to(device), dur)

        pers_loss = PersLoss(y_pred, y_true, cri_pers)
        if train_emb:
            emb_loss = AlignmentLoss(out, cri_align, COSINE, MODS)
            epoch_emb_loss += emb_loss.item()
            loss = alignment_weight * emb_loss + pers_loss
        else: loss = pers_loss
        epoch_pers_loss += pers_loss.item()
        if not evaluate:
            loss.backward()
            opt.step()

    return epoch_emb_loss / (i_batch+1), epoch_pers_loss / (i_batch+1)
