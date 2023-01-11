from utils.datasets import get_dataset
from utils.opts import DATASEED, NUM_LMKS, BASEDIR
import numpy as np
import torch
import os
import random
from scipy.stats import ttest_ind
import pickle

seed = 1024
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)


def get_landmarks(dataset, clfsetting, center_mat, modal,
                  batch_size=8, patch_size=(25, 25, 25), n_threads=8):
    dataloader_list = get_dataset(dataset=dataset, clfsetting=clfsetting, modals=[modal],
                                  patch_size=patch_size, batch_size=batch_size,
                                  center_mat=center_mat, flip_axises=None,
                                  no_smooth=True, no_shuffle=True, no_shift=True,
                                  n_threads=n_threads, seed=DATASEED, resample_patch=None,
                                  trtype='single', only_bl=True)
    train_loader, _, _ = dataloader_list[0]
    # t-test
    patches = []
    groups = []
    for n, data in enumerate(train_loader, 0):
        inputs, _, labels, _ = data

        groups.append(labels[:, 0, 0, 0])
        patches.append(inputs.mean(dim=[-4, -3, -2, -1]))

    patches = torch.cat(patches, dim=0)
    groups = torch.cat(groups, dim=0)

    patches = patches.numpy()
    groups = groups.numpy()

    assert (groups == 1).sum() < 0.5 * (groups.shape[0])
    ad_index = np.where(groups == 1)[0]
    cn_index = np.random.choice(np.where(groups == 0)[0], ad_index.shape[0])

    pvs = []
    for pat_ind in range(patches.shape[1]):
        a = patches[ad_index, pat_ind]
        b = patches[cn_index, pat_ind]
        s, p = ttest_ind(a, b, equal_var=False)
        pvs.append(p)

    pvs = np.array(pvs)
    diff_patinx = pvs.argsort()[:NUM_LMKS]

    DLADLMKS = center_mat[:, diff_patinx]

    with open(os.path.join(BASEDIR, 'utils/DLADLMKS_%d.pkl' % (patch_size[0])), 'wb') as f:
        pickle.dump([DLADLMKS, patch_size, diff_patinx], f)

# if __name__ == '__main__':
#     num_pat = 50
#     PATCH_SIZE = [25, 25, 25]
#
#     opt = parse_opts()
#     opt.dataset = 'ADNI_DX'
#     opt.clfsetting = 'CN-AD'
#     opt.no_shift = True
#     opt.flip_axises = None
#     opt.center_mat = gen_center_mat(PATCH_SIZE)
#     opt.patch_size = PATCH_SIZE
#     opt.batch_size = 8
#     opt.mask_mat = None
#     opt.resample_patch = None
#
#     dataloader_list = get_dataset(dataset=opt.dataset, clfsetting=opt.clfsetting, modal=opt.modal,
#                                   patch_size=opt.patch_size, batch_size=opt.batch_size,
#                                   center_mat=opt.center_mat, flip_axises=opt.flip_axises,
#                                   no_smooth=opt.no_smooth, no_shuffle=opt.no_shuffle, no_shift=opt.no_shift,
#                                   n_threads=opt.n_threads, seed=DATASEED, resample_patch=opt.resample_patch,
#                                   trtype='single', only_bl=True)
