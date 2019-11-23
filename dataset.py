from utils import *
# from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# get frame number from the filename string
def getFrmNo(fnm, format):
    return fnm[fnm.rfind('/')+1:fnm.rfind(f'.{format}')]

# QPS dataset class
class qpDataset(Dataset):
    def __init__(self, mods, df, segs):
        self.mods = mods
        self.samplefun = {'a':self.loadCovarepSample, 'v':self.loadVideoSample,
                          'l':self.loadLangSample}

        self.feat_src = './qps_dataset/' # root directory of the qps dataset

        self.lang_k = 'tencent_emb.npy'
        self.video_k = 'vgg_1fc'

        # under the same clip
        self.segs = segs
        max_n_seg = max([len(seg) for seg in segs])

        self.max_lengths = {'a':44 * (max_n_seg),'v':70 * (max_n_seg),'l':122 * (max_n_seg)}
        self.len = len(self.segs)
        self.meta = df
    def segname(self,seg_id):
        return f'{seg_id:04d}'

    def __len__(self):
        return self.len

    def __getitem__(self,index):
        ed_vote, change, uid, dur_sec = self.meta.loc[self.segs[index][0],['ed_vote', 'change','uid','dur_sec']]
        # print(self.segs[index][0], ed_vote, change)
        sample = {'ed_vote': ed_vote, 'change': change, 'uid':uid, 'dur':dur_sec}

        for mod in self.mods:
            data_window, msk_window = [],[]
            used_segs = self.segs[index]
            for seg_id in used_segs:
                data, msk = self.samplefun[mod](self.segname(seg_id))
                data_window.append(data)
                msk_window.append(msk)
            data = torch.cat(data_window,0)
            msk = torch.cat(msk_window,0)

            # padding
            seq_l = data.size()[0]
            new_sz = list(data.size())
            new_sz[0] = self.max_lengths[mod]
            padded_data = torch.zeros(new_sz, dtype=torch.float)
            padded_data[:seq_l] = data[:seq_l]

            padded_msk = torch.ones(new_sz[0], dtype=torch.bool)
            padded_msk[:seq_l]  = msk[:seq_l]

            sample[f'{mod}_data'] = padded_data
            sample[f'{mod}_msk'] = padded_msk
        return sample

    def loadCovarepSample(self, seg):
        feat = np.load(f'{self.feat_src}/{seg}/covarep_norm.npy')
        return torch.from_numpy(feat), torch.zeros([feat.shape[0]], dtype=torch.bool)

    def loadVideoSample(self, seg):
        # should make masks
        format_k = 'npy'
        imgs = []
        msk = []
        # print(seg)
        fnms = np.sort(np.array(glob.glob(f'{self.feat_src}/{seg}/{self.video_k}/*{format_k}')))
        min_fnm, max_fnm = fnms[0], fnms[-1]
        min_frm, max_frm = int(getFrmNo(min_fnm, format_k)), int(getFrmNo(max_fnm, format_k))

        first_img = None
        for frm in range(min_frm, max_frm + 1):
            # load extracted feats
            fnm = f'{self.feat_src}/{seg}/{self.video_k}/{frm:05}.{format_k}'
            if os.path.isfile(fnm):
                feat = np.load(fnm)
                if first_img is None:
                    first_img = feat
                msk.append(False)
            elif first_img is not None:
                feat = np.zeros_like(first_img)
                msk.append(True)
            imgs.append(torch.from_numpy(feat).unsqueeze(0))

        return torch.cat(imgs, 0), torch.tensor(msk, dtype=torch.bool)

    def loadLangSample(self, seg):
        words = np.load(f'{self.feat_src}/{seg}/{self.lang_k}').astype(np.float32)
        return torch.from_numpy(words),\
               torch.zeros([words.shape[0]], dtype=torch.bool)