import os
import argparse
import pickle as pkl
import torch
import numpy as np
import copy
import logging


DTYPE = torch.float32
BASEDIR = os.path.dirname(os.path.dirname(__file__))
NUM_LMKS = 50

VAL_R = 0.2
TEST_R = 0.2
DATASEED = 1234

LR_PATIENCE = 10
LR_MUL_FACTOR = 0.5

MOMENTUM = 0.9
DAMPENING = 0.9
WEIGHT_DECAY = 1e-3

FULL_PATCH_SIZE = 117, 141, 117
IMG_SIZE = 121, 145, 121
FULL_CENTER_MAT = [[], [], []]
FULL_CENTER_MAT[0].append(int(np.floor((IMG_SIZE[0] - 1) / 2.0)))
FULL_CENTER_MAT[1].append(int(np.floor((IMG_SIZE[1] - 1) / 2.0)))
FULL_CENTER_MAT[2].append(int(np.floor((IMG_SIZE[2] - 1) / 2.0)))
FULL_CENTER_MAT = np.array(FULL_CENTER_MAT)

ADNI_PATH = '/data/datasets/ADNI/ADNI_T1_Fixed_20210719'
AIBL_PATH = '/data/datasets/AIBL'
COBRE_PATH = '/data/datasets/COBRE'

ZIB_AD_path = '/data/datasets/Preprocessing/ZIB_AD/'
ABIDE_T1PATH = '/data/datasets/Preprocessing/ABIDE_T1/'
ABIDE_FMRIPATH = '/data/datasets/Preprocessing/ABIDE_fmri/'
ATLAS_2_PATH='/data/datasets/Preprocessing/ATLAS_2'
MINDS_PATH='/data/datasets/Preprocessing/MINDS'
NIFD_PATH='/data/datasets/Preprocessing/NIFD_T1'
BraTS_PATH='/data/datasets/Preprocessing/MICCAI_BraTS/MICCAI_BraTS_20'
eQTL_filtering = 'ZIB_snp_filtered'
eQTL_PATH = os.path.join('/data/home/zhangzc/dataset/brain_imgen/gtex_link.eqtl.filtered', eQTL_filtering)

# logging.warning('GO Graph path: %s' % eQTL_PATH)

# GO_GRAPH_PATH = os.getenv('GO_GRAPH_PATH', os.path.join(eQTL_PATH, 'go_graph'))
GO_GRAPH_PATH = os.getenv('GO_GRAPH_PATH', os.path.join(eQTL_PATH, 'go_graph_flattened'))
# logging.warning('GO Graph path: %s' % GO_GRAPH_PATH)


def gen_center_mat(pat_size: list):
    center_mat = [[], [], []]
    for x in np.arange(pat_size[0] // 2, IMG_SIZE[0] // pat_size[0] * pat_size[0], pat_size[0]):
        for y in np.arange(pat_size[1] // 2, IMG_SIZE[1] // pat_size[1] * pat_size[1], pat_size[1]):
            for z in np.arange(pat_size[2] // 2, IMG_SIZE[2] // pat_size[2] * pat_size[2], pat_size[2]):
                center_mat[0].append(x + (IMG_SIZE[0] % pat_size[0]) // 2)
                center_mat[1].append(y + (IMG_SIZE[1] % pat_size[1]) // 2)
                center_mat[2].append(z + (IMG_SIZE[2] % pat_size[2]) // 2)
    center_mat = np.array(center_mat)
    return center_mat


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='ADNI_DX',
        type=str,
        help='Select dataset, MIRIAD|AIBL|OASIS|HABS is for testing only.'
             ' (ADNI_PETdx|ADNI_dx|ADNI_PET_DX|ADNI_PET|MIRIAD|AIBL|OASIS|HABS)')

    parser.add_argument(
        '--pre_datasets',
        default=None,
        nargs='+',
        required=False,
        help='dataset list for prediction (Optional)')

    parser.add_argument(
        '--label_names',
        default=['FDG', 'AV45'],
        type=str,
        nargs='+',
        help='regression lables (FDG, AV45)')

    parser.add_argument(
        '--modals',
        default=['mwp1'],
        type=str,
        nargs='+',
        help='modalities of MRI (mwp1|NATIVE_GM_|wm)')

    parser.add_argument(
        '--clfsetting',
        default='CN-AD',
        type=str,
        help='classification setting (regression|CN-AD|CN_sMCI-pMCI_AD|sMCI-pMCI)')

    parser.add_argument(
        '--trtype',
        default='single',
        type=str,
        help='training type, single|5-rep')

    parser.add_argument(
        '--method',
        default='ViT',
        type=str,
        help=
        'choose a method.'
    )
    parser.add_argument(
        '--batch_size',
        default=5,
        type=int,
        help='Batch Size')

    parser.add_argument(
        '--patch_size',
        default=(25, 25, 25),
        type=int,
        nargs=3,
        help='patch size, only available for some methods')

    parser.add_argument(
        '--pretrain_phase',
        default=0,
        type=int,
        help='pretrain_phase')

    parser.add_argument(
        '--optimizer',
        default='adam',
        type=str,
        help=
        'Optimizer, adam|sgd')

    parser.add_argument(
        '--cuda_index',
        default=0,
        type=int,
        help='Specify the index of gpu')

    parser.add_argument(
        '--global_epochs',
        default=1,
        type=int,
        help='Number of total epochs to run')

    parser.add_argument(
        '--learning_rate',
        default=1e-4,
        type=float,
        help=
        'Initial learning rate (divided by factor while training by lr scheduler)')

    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)

    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)

    parser.add_argument(
        '--no_log',
        action='store_true',
        help='If true, tensorboard logging is not used.')
    parser.set_defaults(no_log=False)

    parser.add_argument(
        '--gen_lmk',
        action='store_true',
        help='If true, landmarks for CN vs. AD classification will be generated.')
    parser.set_defaults(gen_lmk=False)

    parser.add_argument(
        '--n_threads',
        default=8,
        type=int,
        help='Number of threads for multi-thread loading')

    # PRE_TRAIN_MODEL = 'saved_model/Oct16_13-01-29_vgpu02Resnet34Sig_NormedGM_NoOverlap_FullImg_batch4.pth'
    parser.add_argument(
        '--pretrain_path',
        default=None,
        type=str,
        help='Pretrained model (.pth)')

    parser.add_argument(
        '--log_label',
        default='',
        type=str,
        help='additional label for logging')

    parser.add_argument(
        '--test',
        action='store_true',
        help='If true, test is performed.')
    parser.set_defaults(test=False)

    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)

    parser.add_argument(
        '--flip_axises',
        default=[0, 1, 2],
        type=int,
        nargs='+',
        help='flip axises (0, 1, 2)')

    parser.add_argument(
        '--no_smooth',
        action='store_true',
        help='no smooth apply to MRI')
    parser.set_defaults(no_smooth=False)

    parser.add_argument(
        '--no_shift',
        action='store_true',
        help='no shift apply to MRI for data augmentation')
    parser.set_defaults(no_shift=False)

    parser.add_argument(
        '--no_shuffle',
        action='store_true',
        help='no shuffle apply to batch sampling')
    parser.set_defaults(no_shuffle=False)

    parser.add_argument(
        '--method_para',
        default='{}',
        type=str,
        help='specify method parameters in dict form. eg: {"para1": values1}')

    parser.add_argument(
        '--para_search',
        action='store_true',
        help='apply parameter searching')
    parser.set_defaults(para_search=False)

#federated arguments
    parser.add_argument(
        '--federated_para',
        default='{}',
        type=str,
        help='specify federated parameters in dict form. eg: {"para1": values1}'
    )

    parser.add_argument(
        '--federated',
        action='store_true',
        help='federated learning')
    parser.set_defaults(federated=False)

    parser.add_argument(
        '--federated_setting',
        default=None,
        type=str,
        help='specify federated parameters in dict form. eg: {"para1": values1}'
    )

    parser.add_argument(
        '--withDP',
        action='store_true',
        help='diffential privacy'
    )

    parser.add_argument(
        '--dp_setting',
        default=None,
        type=str,
        help='specify dp parameters in dict form. eg: {"para1": values1}'
    )

    args = parser.parse_args()
    return args

def mod_opt(method, opt):
    opt = copy.copy(opt)
    opt.mask_mat = None
    opt.resample_patch = None

    if method in ['Res18', 'Res34']:
        opt.center_mat = FULL_CENTER_MAT
        opt.patch_size = FULL_PATCH_SIZE
        opt.method_setting = {'sample_size': opt.patch_size, 'num_classes': opt.num_classes}

    elif method in ['Res18_Multimodal']:
        opt.center_mat = FULL_CENTER_MAT
        opt.patch_size = FULL_PATCH_SIZE
        opt.method_setting = {'sample_size': opt.patch_size, 'num_classes': opt.num_classes}

    elif method in ['densenet121']:
        opt.center_mat = FULL_CENTER_MAT
        opt.patch_size = FULL_PATCH_SIZE
        opt.method_setting = {'sample_size1': opt.patch_size[1], 'sample_size2':opt.patch_size[2],
                              'sample_duration':opt.patch_size[0],'num_classes': opt.num_classes}
    elif method in ['WideResNet18']:
        opt.center_mat = FULL_CENTER_MAT
        opt.patch_size = FULL_PATCH_SIZE
        opt.method_setting = {'sample_size1': opt.patch_size[1], 'sample_size2':opt.patch_size[2],
                              'sample_duration':opt.patch_size[0],'num_classes': opt.num_classes}# TODO: WideResnet18,34

    elif method in ['WideResNet34']:
        opt.center_mat = FULL_CENTER_MAT
        opt.patch_size = FULL_PATCH_SIZE
        opt.method_setting = {'sample_size1': opt.patch_size[1], 'sample_size2': opt.patch_size[2],
                              'sample_duration': opt.patch_size[0], 'num_classes': opt.num_classes}

    elif method in ['PreActivationResNet18','PreActivationResNet34']:
        opt.center_mat = FULL_CENTER_MAT
        opt.patch_size = FULL_PATCH_SIZE
        opt.method_setting = {'sample_size1': opt.patch_size[1], 'sample_size2':opt.patch_size[2],
                              'sample_duration':opt.patch_size[0],'num_classes': opt.num_classes}

    elif method == 'DAMIDL':
        with open('/data/home/feiyl/202301/utils/DLADLMKS_%d.pkl' % opt.patch_size[0], 'rb') as f:
            opt.center_mat, opt.patch_size, _ = pkl.load(f)
        opt.center_mat = opt.center_mat[:, :NUM_LMKS]
        if not opt.patch_size[0] == 25:
            opt.resample_patch = [25, 25, 25]
        opt.method_setting = {
            'patch_num': opt.center_mat.shape[1], 'feature_depth': [32, 64, 128, 128]
        }

    elif method == 'ViT':
        opt.center_mat = gen_center_mat(opt.patch_size)
        opt.method_setting = {
            'patch_size': opt.patch_size, 'num_patches': opt.center_mat.shape[1],
            'dim': 64, 'depth': 4, 'heads': 8, 'dim_head': 64, 'mlp_dim': 4 * 64,
            'dropout': 0.1, 'emb_dropout': 0.1, 'num_classes': 1, 'pool': 'cls', 'channels': 1}

    else:
        raise NotImplementedError
    return opt

def federated_opt(opt):
    opt = copy.copy(opt)
    opt.federated_setting={
        'num_users':4,'frac':0.5,'local_bs':1,'iid':0,'unequal':1,'local_epoch':5, 'fed_batch_size':1, 'seperate_test_data':1
    }
    return opt
    #'iid':set to 0 for non-IID
    #'unequal':set to 0 for equal splits

def dp_opt(opt):
    opt = copy.copy(opt)
    opt.dp_setting={
        'delta':1e-3, 'epsilon':4.0, 'max_grad_norm':2
    }
    return opt
