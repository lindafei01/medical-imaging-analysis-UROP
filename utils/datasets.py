import logging
import datetime
logging.basicConfig(level='WARNING')
import torch
import numpy as np
import os
import torch.utils.data as data
from .opts import DTYPE, BASEDIR, VAL_R, TEST_R
import re
import pickle as pkl
from nilearn.image import smooth_img
import nibabel as nib
import pandas as pd
from scipy import ndimage as nd
from sklearn.model_selection import train_test_split
from typing import Union, Tuple, Iterable
from torch.utils.data import Dataset
import scipy.io

# ABIDE_PATH='/data/datasets/Preprocessing/ABIDE_T1'
# ATLAS_2_PATH='/data/datasets/Preprocessing/ATLAS_2'
# MINDS_PATH='/data/datasets/Preprocessing/MINDS'
# NIFD_PATH='/data/datasets/Preprocessing/NIFD_T1'
# BraTS_PATH='/data/datasets/Preprocessing/MICCAI_BraTS/MICCAI_BraTS_20'
from .opts import ADNI_PATH, AIBL_PATH, COBRE_PATH, eQTL_PATH, ZIB_AD_path, ABIDE_T1PATH, ABIDE_FMRIPATH, ATLAS_2_PATH, MINDS_PATH, NIFD_PATH, BraTS_PATH


def batch_sampling(imgs, labels, center_mat, aux_labels, dis_labels, patch_size=(25, 25, 25), random=False,
                   shift=False, flip_axis=None):
    shift_range = [-2, -1, 0, 1, 2]
    flip_pro = 0.3
    num_patch = len(center_mat[0])
    batch_size = len(imgs)
    margin = [int(np.floor((i - 1) / 2.0)) for i in patch_size]

    batch_img = torch.tensor(data=np.zeros([num_patch * batch_size, 1, patch_size[0], patch_size[1], patch_size[2]]),
                             dtype=DTYPE)
    batch_label = torch.tensor(data=np.zeros([num_patch * batch_size] + list(labels.shape[1:])), dtype=DTYPE)
    batch_aux_label = torch.tensor(data=np.zeros([num_patch * batch_size] + list(aux_labels.shape[1:])), dtype=DTYPE)
    batch_dis_label = torch.tensor(data=np.zeros([num_patch * batch_size] + list(dis_labels.shape[1:])), dtype=DTYPE)

    for num, data in enumerate(zip(imgs, labels, aux_labels, dis_labels), start=0):
        img, label, aux_label, dis_label = data
        if not random:
            for ind, cors in enumerate(zip(center_mat[0], center_mat[1], center_mat[2])):
                x_cor, y_cor, z_cor = cors
                if shift:
                    x_scor = x_cor + shift_range[torch.randint(high=len(shift_range), size=(1,))]
                    y_scor = y_cor + shift_range[torch.randint(high=len(shift_range), size=(1,))]
                    z_scor = z_cor + shift_range[torch.randint(high=len(shift_range), size=(1,))]
                else:
                    x_scor, y_scor, z_scor = x_cor, y_cor, z_cor

                single_patch = img[:,
                               max(x_scor - margin[0], 0): x_scor + margin[0] + 1,
                               max(y_scor - margin[1], 0): y_scor + margin[1] + 1,
                               max(z_scor - margin[2], 0): z_scor + margin[2] + 1]
                if (not (flip_axis is None)) and (torch.rand(1) < flip_pro):
                    if isinstance(flip_axis, list):
                        single_patch = single_patch.flip(flip_axis[torch.randint(high=len(flip_axis), size=(1,))])
                    single_patch = single_patch.flip(flip_axis)

                batch_img[ind + num * num_patch,
                :single_patch.shape[0],
                :single_patch.shape[1],
                :single_patch.shape[2],
                :single_patch.shape[3]] = single_patch

                # batch_img[ind + num * num_patch] = single_patch

                batch_label[ind + num * num_patch] = label
                batch_aux_label[ind + num * num_patch] = aux_label
                batch_dis_label[ind + num * num_patch] = dis_label
        else:
            raise NotImplementedError

    return batch_img, batch_aux_label, batch_label, batch_dis_label

class Dataset(object):
    def __init__(self, image_path, subset, clfsetting, modals: list, no_smooth=False,
                 only_bl=False, ids_exclude=(), ids_include=(),
                 maskmat=None, seed=1234, preload=True, dsettings=None):
        '''Meta class for every dataset
        should at least implement self.get_table, self._filtering, self.dx_mapping,
        '''

        if dsettings is None:
            dsettings = {}
        self.preload = preload
        self.image_path = image_path
        self.smooth = not no_smooth
        self.maskmat = maskmat
        self.clfsetting = clfsetting

        info = self.get_table(self.image_path, modals, dsettings)
        info['GROUP'] = self.g_mapping(info['DX'].values)

        for col in ['DX', 'ID', 'GROUP', 'VISIT'] + modals:
            assert col in info.columns

        # subject level selection
        info = info[info['ID'].isin(self._filtering(info))]
        if len(ids_include):
            info = info[info['ID'].isin(ids_include)]
        if len(ids_exclude):
            info = info[~info['ID'].isin(ids_exclude)]

        # entry level selection
        # info['ID'] = info['ID'].astype(int)
        info['VISIT'] = info['VISIT'].values.astype(int)
        info = info.sort_values('VISIT')

        info = info[~info[modals[0]].isnull()]  # exclude the entry without img
        # *note*: if a subject got several DX in diff time point, then only use the earliest DX (should be sorted first)
        # this avoid using the same subject in different clfsettings.

        if clfsetting=='DIS-NC':
            info['DX_LABEL'] = self.DIS_NC_mapping(info['DX'].values, clfsetting)
            for i in np.sort(info['ID'].unique()):  # using -1 to represent the unavailable dx
                dx = info.loc[info['ID'] == i, 'DX_LABEL'].values
                dx = dx[~np.isnan(dx)]
                if dx[0] == -1:
                    info.loc[info['ID'] == i, 'DX_LABEL'] = -1
        else:
            info['DX_LABEL'] = self.dx_mapping(info['DX'].values, clfsetting)
            for i in np.sort(info['ID'].unique()):  # using -1 to represent the unavailable dx
                dx = info.loc[info['ID'] == i, 'DX_LABEL'].values
                dx = dx[~np.isnan(dx)]
                if dx[0] == -1:
                    info.loc[info['ID'] == i, 'DX_LABEL'] = -1

        labels = self.get_label(info, dsettings)
        inx = ~np.isnan(labels).all(axis=-1).all(axis=-1)
        inx &= labels[:, -1, -1] != -1  # exclusion for dx == -1
        labels = labels[inx]
        info = info[inx]

        if only_bl:
            inx = ~info.duplicated('ID')
            labels = labels[inx]
            info = info[inx]

        train_inx, val_inx, test_inx = self.data_split(stratify_var=info['GROUP'].values, id_info=info['ID'].values,
                                                       seed=seed)
        inx = []
        if 'training' in subset:
            inx = np.append(inx, train_inx)
        if 'validation' in subset:
            inx = np.append(inx, val_inx)
        if 'testing' in subset:
            inx = np.append(inx, test_inx)
        inx = inx.astype(int)

        self.labels = labels[inx]  # shape for classification setting: (n, 1, 1)
        self.dis_label = info['GROUP'].values[inx]  # shape : (n,)
        self.id_info = info['ID'].values[inx]  # shape : (n)
        self.namelist = info[modals[0]].values[inx]  # shape : (n,)
        self.aux_labels = np.array([info[modal].values[inx] for modal in modals[1:]]) if modals[1:] else None
        # shape : (n, k, *), k: len(modals) - 1,

        self.info = info.iloc[inx]  # shape : (n, c)

        self.data = [None for i in self.namelist]

    def get_table(self, path, modals: Union[list, str], dsettings: Union[None, dict]) -> pd.DataFrame:
        '''return the pd.DataFrame contains all needed information, which should at least include following columns:
            ['DX', 'ID', 'VISIT'] + modals

        Args:
            path: root path for the dataset
            modals: modalities to use
            dsettings: specific settings for the dataset

        Returns:
            pd.DataFrame

        '''
        raise NotImplementedError

    @staticmethod
    def DIS_NC_mapping(dx, clfsetting):
        if clfsetting == 'DIS-NC':
            binary_mapping={np.nan: -1, '': -1, 'nan': -1,
                            'CN': 0, 'NC': 0, 'No_Known_Disorder': 0, 'Control': 0, 'Healthy Control': 0,
                            'sMCI': 1, 'pMCI': 1, 'AD': 1,
                            'Bipolar_Disorder': 1, 'Bipolar Disorder': 1,
                            'Schizoaffective': 1, 'Schizophrenia_Strict': 1, 'Schizophrenia': 1,
                            'Autism': 1,
                            'HGG': 1, 'LGG': 1, 'unknown Tumor': 1,
                            'Stroke': 1, 'Major Depressive Disorder': 1, 'FTD': 1}
            labels = map(lambda x: binary_mapping[x], dx)
            labels = np.array(list(labels), dtype=int)

        else:
            raise NotImplementedError

        return labels

    @staticmethod
    def _filtering(info: pd.DataFrame) -> Union[list, tuple, set]:
        '''filtering for each dataset

        Args:
            info:  pd.DataFrame, contains DX, ID, VISIT columns

        Returns:
            a list for ids that should be included in the dataset
        '''
        raise NotImplementedError

    @staticmethod
    def dx_mapping(dx: Iterable, clfsetting: str) -> np.ndarray:
        '''mapping DX

        Args:
            dx: shape: (n,), str type diagnoses
            clfsetting: classification setting

        Returns:
            np.ndarray with shape of (n,) and dtype of int
        '''
        raise NotImplementedError

    @staticmethod
    def g_mapping(dx: Iterable):
        '''mapping str diagnosis to int code

        Args:
            dx: shape: (n, ), str type diagnoses

        Returns:
            np.ndarray with shape of (n,) and dtype of int
        '''
        dis_map = {np.nan: -1, '': -1, 'nan': -1,
                   'CN': 0, 'NC': 0, 'No_Known_Disorder': 0,'Control':0,'Healthy Control':0,
                   'sMCI': 1, 'pMCI': 2, 'AD': 3,
                   'Bipolar_Disorder': 4, 'Bipolar Disorder':4,
                   'Schizoaffective': 5, 'Schizophrenia_Strict': 6,'Schizophrenia':6,
                   'Autism':7,
                   'HGG':8,'LGG':9,'unknown Tumor':10,
                   'Stroke':11,
                   'Major Depressive Disorder':12,'FTD':13}
        gcode = np.array(list(map(lambda x: dis_map[x], dx)))
        return gcode

    @staticmethod
    def get_label(info: pd.DataFrame, dsettings: dict) -> np.ndarray:
        '''return labels for training

        Args:
            info:
            dsettings:

        Returns:
            labels: Has a shape of (n, 1, 1) for classification.
        '''
        if dsettings == {}:
            labels = info['DX_LABEL'].values.reshape([-1, 1, 1])
        else:
            raise NotImplementedError

        return labels

    @staticmethod
    def data_split(stratify_var: np.ndarray, id_info: np.ndarray, seed: int):
        '''return the index for training set, validation set, and testing set,
        where each subset do not have subject level overlapping

        Args:
            stratify_var: int type, shape: (n,). Data is split in a stratified fashion, using this as
        the class labels.

            id_info: shape: (n,) the id for each subject
            seed: random seed for reproduction

        Returns:
            train_inx
            val_inx
            test_inx
        '''
        stratify = stratify_var
        train_inx, val_test_inx = train_test_nooverlap(len(id_info), test_size=VAL_R + TEST_R,
                                                       id_info=id_info, seed=seed,
                                                       stratify=stratify)
        val_inx, test_inx = train_test_nooverlap(len(val_test_inx), test_size=VAL_R / (VAL_R + TEST_R),
                                                 id_info=id_info[val_test_inx], seed=seed + 10000,
                                                 stratify=stratify[val_test_inx])
        val_inx = np.array(val_test_inx)[val_inx].tolist()
        test_inx = np.array(val_test_inx)[test_inx].tolist()

        return train_inx, val_inx, test_inx

    @staticmethod
    def check_imgpath(subj_path, modal):
        if modal in ['mwp1']:
            if os.path.exists(os.path.join(subj_path, 'report', 'catreport_T1w.pdf')) or os.path.exists(os.path.join(subj_path, 'report', 'catreport_T1W.pdf')):
                if os.path.exists(os.path.join(subj_path, 'mri', modal + 'T1w.nii')):
                    return os.path.join(subj_path, 'mri', modal + 'T1w.nii')
                if os.path.exists(os.path.join(subj_path, 'mri', modal + 'T1W.nii')):
                    return os.path.join(subj_path, 'mri', modal + 'T1W.nii')
            else:
                return None
        else:
            raise NotImplementedError

    def load_data(self, img_path, name, smooth=False) -> np.ndarray:
        '''load MRI data from disk

        Args:
            img_path:
            name:
            smooth: if Ture, Smooth MRI images by applying a Gaussian filter with kernel size of 8.

        Returns:
            ori_img: the MRI data matrix in which nan is set to 0.
        '''
        brain_label = None
        if 'NATIVE_GM_' in name:
            dir, file = os.path.split(os.path.join(img_path, name))
            file = file.replace('NATIVE_GM_', '')
            ori_img = nib.load(os.path.join(dir, '../', file))
            brain_label = nib.load(os.path.join(dir, 'p0' + file)).get_fdata()
        elif 'NATIVE_WM_' in name:
            dir, file = os.path.split(os.path.join(img_path, name))
            file = file.replace('NATIVE_GM_', '')
            ori_img = nib.load(os.path.join(dir, '../', file))
            brain_label = nib.load(os.path.join(dir, 'p0' + file)).get_fdata()
        else:
            ori_img = nib.load(os.path.join(img_path, name))

        if smooth:
            ori_img = smooth_img(ori_img, 8).get_fdata()
        else:
            ori_img = ori_img.get_fdata()

        if 'NATIVE_GM_' in name:
            ori_img[brain_label < 1.5] = 0
            ori_img[brain_label >= 2.5] = 0
        elif 'NATIVE_WM_' in name:
            ori_img[brain_label < 2.5] = 0

        ori_img[np.isnan(ori_img)] = 0
        return ori_img

    def __getitem__(self, index):
        if self.preload:
            if self.data[index] is not None:
                bat_data = self.data[index]
            else:
                name = self.namelist[index]
                if self.maskmat is None:
                    bat_data = (self.load_data(self.image_path, name, self.smooth))
                else:
                    bat_data = (self.load_data(self.image_path, name, self.smooth) * self.maskmat)
                bat_data = np.array(bat_data, dtype=np.float32)
                bat_data[np.isnan(bat_data)] = 0
                self.data[index] = bat_data
        else:
            name = self.namelist[index]
            if self.maskmat is None:
                bat_data = (self.load_data(self.image_path, name, self.smooth))
            else:
                bat_data = (self.load_data(self.image_path, name, self.smooth) * self.maskmat)
            bat_data = np.array(bat_data, dtype=np.float32)
            bat_data[np.isnan(bat_data)] = 0

        bat_data = torch.from_numpy(bat_data).unsqueeze(0).unsqueeze(0)  # channel, batch
        bat_labels = torch.Tensor(np.array([self.labels[index]]))
        if self.aux_labels is not None:
            if (self.aux_labels.dtype in [float, np.float16, np.float32, np.float64]):
                bat_aux_label = torch.Tensor([self.aux_labels[index]])
            else:
                raise NotImplementedError
        else:
            bat_aux_label = bat_labels * torch.from_numpy(np.array([np.nan]))

        if self.dis_label is not None:
            bat_dis_label = torch.Tensor([self.dis_label[index]])
        else:
            bat_dis_label = torch.from_numpy(np.array([np.nan]))

        return bat_data, bat_labels, bat_aux_label, bat_dis_label

    @staticmethod
    def load_fmridata(modal, path = ABIDE_FMRIPATH) -> pd.DataFrame :

        path = os.path.join(path,'ABIDE_pred')
        valid_subjects = list(filter(lambda x: (os.path.exists(
            os.path.join(path, x, 'conn_processing/results/firstlevel/SBC_01',
                         'resultsROI_Subject001_Condition001.mat'))), sorted(os.listdir(path))))
        Z_vectors = []
        #Z_vectors_Premove = np.empty(shape=(0, len(P_index)))

        for subject in valid_subjects:

            Z_map = np.array(scipy.io.loadmat(os.path.join(path, subject, 'conn_processing/results/firstlevel/SBC_01',
                                                           'resultsROI_Subject001_Condition001.mat'))['Z'])
            array_size = ((Z_map.shape[0] - 1) * Z_map.shape[0]) // 2
            #array_size = ((Z_map.shape[0] - 34) * (Z_map.shape[0] - 33) ) // 2
            #array_size = 33 * 32 //2
            Z_vector = np.zeros(shape=(1, array_size))
            pos1 = 0
            for row in range(Z_map.shape[0]):
                for column in range(row):
                        Z_vector[0, pos1] = Z_map[row, column]
                        pos1 += 1

            Z_vectors.append(Z_vector)

        Z_dict = { modal: Z_vectors,'ID': valid_subjects}
        Z_dataframe = pd.DataFrame(Z_dict)

        return Z_dataframe

    def __add__(self, other):
        assert isinstance(other, Dataset)
        assert self.preload == other.preload
        assert self.smooth == other.smooth
        assert self.maskmat == other.maskmat
        assert self.clfsetting == other.clfsetting
        assert type(self.aux_labels) == type(other.aux_labels)

        new_ins = Dataset.__new__(Dataset)
        new_ins.preload = self.preload
        new_ins.smooth = self.smooth
        new_ins.maskmat = self.maskmat
        new_ins.clfsetting = self.clfsetting

        new_ins.labels = np.append(self.labels, other.labels, axis=0)
        new_ins.dis_label = np.append(self.dis_label, other.dis_label, axis=0)
        new_ins.namelist = np.append(self.namelist, other.namelist, axis=0)
        new_ins.info = self.info.append(other.info)
        new_ins.id_info = np.append(self.id_info, other.id_info, axis=0)

        if self.aux_labels is None:
            new_ins.aux_labels = None
        else:
            new_ins.aux_labels = np.append(self.aux_labels, other.aux_labels)

        new_ins.data = self.data + other.data

        return new_ins

    def __len__(self):
        return len(self.id_info)

class Patch_Data(data.Dataset):
    def __init__(self, imgdata, patch_size, center_mat, shift, flip_axis, resample_patch=None):
        self.patch_size = patch_size
        self.center_mat = center_mat
        self.shift = shift
        self.flip_axis = flip_axis
        self.imgdata = imgdata
        self.labels = imgdata.labels
        self.resample_patch = resample_patch

    def __getitem__(self, index):
        bat_data, bat_labels, bat_aux_label, bat_dis_label = self.imgdata[index]
        inputs, aux_labels, labels, dis_label = batch_sampling(imgs=bat_data, labels=bat_labels,
                                                               center_mat=self.center_mat,
                                                               aux_labels=bat_aux_label, dis_labels=bat_dis_label,
                                                               patch_size=self.patch_size,
                                                               random=False, shift=self.shift,
                                                               flip_axis=self.flip_axis,
                                                               )

        if self.resample_patch is not None:
            assert len(self.resample_patch) == 3
            _inputs = inputs.numpy()
            resam = np.zeros(list(inputs.shape[:-3]) + self.resample_patch)
            dsfactor = [w / float(f) for w, f in zip(self.resample_patch, _inputs.shape[-3:])]
            for i in range(_inputs.shape[0]):
                resam[i, 0, :] = nd.interpolation.zoom(_inputs[i, 0, :], zoom=dsfactor)
            inputs = torch.from_numpy(resam)

        return inputs.squeeze(0), aux_labels.squeeze(0), labels.squeeze(0), dis_label.squeeze(0),

    def __len__(self):
        return len(self.imgdata)

def train_test_nooverlap(data_size, test_size, id_info, seed=None, stratify=None):
    test = []
    train = []
    inxes = np.arange(data_size)

    if stratify is not None:
        df = pd.DataFrame([id_info.ravel(), stratify.ravel()]).transpose()
        df.columns = ['id', 'stf']
        ids_set, stratify = df.drop_duplicates('id').values.T
        sort_inx = ids_set.argsort()
        stratify = stratify.astype(np.int)[sort_inx]
        ids_set = ids_set[sort_inx]
    else:
        ids_set = np.unique(id_info)
        ids_set = np.sort(ids_set)  # remove randomness induced by memory stuff

    train_ids, test_ids = train_test_split(ids_set, test_size=test_size,
                                           random_state=seed, shuffle=True, stratify=stratify)

    for id in test_ids:
        test += (inxes[id_info == id]).tolist()
    for id in train_ids:
        train += (inxes[id_info == id]).tolist()

    return train, test

class ADNI(Dataset):
    # TODO: normalization for regression
    def __init__(self, subset, clfsetting, modals, no_smooth=False, only_bl=False,
                 ids_exclude=(), ids_include=(), maskmat=None, seed=1234, preload=True,
                 dsettings=None):

        if dsettings is None:
            dsettings = {'cohort': 'ALL', 'label_names': ['DX']}
        image_path = ADNI_PATH
        super(ADNI, self).__init__(image_path, subset, clfsetting, modals, no_smooth,
                                   only_bl, ids_exclude, ids_include, maskmat, seed, preload, dsettings)

    @staticmethod
    def get_csf():
        csf_info = pd.read_csv(os.path.join(BASEDIR, 'data/UPENNBIOMK9_MERGE_AUX_20201113.csv'), dtype=str)
        csf_info = csf_info[~csf_info['ABETA'].isna()]
        csf_info = csf_info[~csf_info['TAU'].isna()]
        csf_info = csf_info[~csf_info['PTAU'].isna()]
        for i in csf_info.index:
            if '>' in csf_info.loc[i]['ABETA']:
                if 'Recalculated' in csf_info.loc[i]['COMMENT']:
                    csf_info.loc[i]['ABETA'] = re.search('^Recalculated ABETA result = ([0-9]{1,}) pg/mL$',
                                                         csf_info.loc[i]['COMMENT']).group(1)
                else:
                    csf_info.loc[i]['ABETA'] = csf_info.loc[i]['ABETA'].replace('>', '')
        csf_info = csf_info[~csf_info['ABETA'].str.contains('<')]
        csf_info = csf_info[~csf_info['TAU'].str.contains('<')]
        csf_info = csf_info[~csf_info['TAU'].str.contains('>')]
        csf_info = csf_info[~csf_info['PTAU'].str.contains('<')]
        csf_info = csf_info[~csf_info['PTAU'].str.contains('>')]
        csf_info = csf_info.drop('VISCODE_x', axis=1)
        csf_info.rename(columns={'EXAMDATE': 'EXAMDATE_CSF', 'VISCODE2': 'VISCODE'}, inplace=True)
        # tau_id = set([i for i in csf_info[~csf_info['TAU'].isna()]['RID'].to_list()])
        # abeta_id = set([i for i in csf_info[~csf_info['ABETA'].isna()]['RID'].to_list()])
        return csf_info[['RID', 'VISCODE', 'EXAMDATE_CSF', 'ABETA', 'TAU', 'PTAU']]

    @staticmethod
    def get_fdg():
        UCB_FDG = pd.read_csv(os.path.join(BASEDIR, 'data/UCBERKELEYFDG_05_28_20.csv'), dtype=str)
        UCB_FDG.rename(columns={'EXAMDATE': 'EXAMDATE_FDG', }, inplace=True)
        # Consider PUTAMEN
        rois = [["Angular", "Left"],
                ["Angular", "Right"],
                ["CingulumPost", "Bilateral"],
                ["Temporal", "Left"],
                ["Temporal", "Right"], ]
        ROI_FDG = []
        UCB_FDG = UCB_FDG[~pd.isnull(UCB_FDG['VISCODE2'])]

        # problematic data(viscode is wrong and duplicated)
        UCB_FDG = UCB_FDG[UCB_FDG['RID'] != '4765']

        UCB_FDG['VISCODE_int'] = UCB_FDG['VISCODE2'].apply(lambda x: 0 if x == 'bl' else int(x.replace('m', '')))

        UCB_FDG = UCB_FDG.sort_values(by=['RID', 'VISCODE_int'], kind='mergesort')
        ind_df = UCB_FDG[['RID', 'VISCODE2', 'EXAMDATE_FDG']].drop_duplicates()
        for roi, l in rois:
            # TODO: sorting
            temp = UCB_FDG[(UCB_FDG['ROINAME'] == roi) & (UCB_FDG['ROILAT'] == l)]
            ROI_FDG.append(temp['MEAN'].values)
            assert (temp[['RID', 'VISCODE2']].values == ind_df[['RID', 'VISCODE2']].values).all()
        ROI_FDG = pd.DataFrame(np.array(ROI_FDG).T, columns=['_'.join(i) for i in rois])
        ROI_FDG['RID'] = ind_df.values[:, 0]
        ROI_FDG['VISCODE'] = ind_df.values[:, 1]
        ROI_FDG['FDG_SUMMARY'] = ROI_FDG[['_'.join(i) for i in rois]].astype(float).values.mean(axis=1)
        return ROI_FDG

    @staticmethod
    def get_av45():
        UCB_AV45 = pd.read_csv(os.path.join(BASEDIR, 'data/UCBERKELEYAV45_01_14_21.csv'), dtype=str)
        UCB_AV45 = UCB_AV45.drop('VISCODE', axis=1)
        UCB_AV45 = UCB_AV45.drop('update_stamp', axis=1)
        UCB_AV45.rename(columns={'EXAMDATE': 'EXAMDATE_AV45', 'VISCODE2': 'VISCODE'}, inplace=True)

        ROI_AV45 = UCB_AV45[['RID', 'VISCODE', 'EXAMDATE_AV45', 'FRONTAL_SUVR', 'CINGULATE_SUVR', 'PARIETAL_SUVR',
                             'TEMPORAL_SUVR', 'COMPOSITE_REF_SUVR', 'SUMMARYSUVR_COMPOSITE_REFNORM',
                             'SUMMARYSUVR_COMPOSITE_REFNORM_0.78CUTOFF']]

        return ROI_AV45

    @staticmethod
    def get_info():
        info = pd.read_csv(os.path.join(BASEDIR, 'data/ADNIMERGE_MODIFIED_20201227.csv'), dtype=str)
        info.loc[(info['RID'] == '2') & (info['VISCODE'] == 'm90'), 'EXAMDATE'] = '2013/3/2'
        return info[['RID', 'VISCODE', 'COLPROT', 'EXAMDATE', ]]

    @staticmethod
    def get_aux():
        info = pd.read_csv(os.path.join(BASEDIR, 'data/ADNIMERGE_MODIFIED_20201227.csv'), dtype=str)
        demo_info = pd.read_csv(os.path.join(BASEDIR, 'data/PTDEMOG.csv'), dtype=str)
        info.loc[(info['RID'] == '2') & (info['VISCODE'] == 'm90'), 'EXAMDATE'] = '2013/3/2'
        # add birth Year Month
        info = pd.merge(how='left', left=info, right=demo_info[['RID', 'PTDOBYY']].drop_duplicates().dropna(),
                        on=['RID'])
        info['PTGENDER'] = info['PTGENDER'].values.astype(int)
        info['AGE'] = (pd.to_datetime(info['EXAMDATE']).dt.year - pd.to_datetime(info['PTDOBYY']).dt.year)
        info.rename(columns={'EXAMDATE': 'EXAMDATE_AUX'}, inplace=True)
        info = info[['RID', 'VISCODE', 'EXAMDATE_AUX',
                     'AGE', 'PTGENDER', 'APOE4', 'mPACCdigit', 'ADASQ4', 'Hippocampus', 'MMSE']]
        info['AGE'] = info['AGE'].astype(float) / 100.
        info['ADASQ4'] = info['ADASQ4'].astype(float) / 10.

        return info

    @staticmethod
    def get_dx():
        info = pd.read_csv(os.path.join(BASEDIR, 'data/ADNIMERGE_MODIFIED_20201227.csv'), dtype=str)
        # dx_df = pd.read_csv(os.path.join(BASEDIR, 'data/DXSUM_PDXCONV_ADNIALL_20220418.csv'), dtype=str)
        # subjects that is not AD type dementia TODO: confirm
        info = info[~info['RID'].isin(['93', '769'])]
        info['NEW_DX'] = info[pd.isnull(info['NEW_DX'])]['NEW_DX'].drop_duplicates()  # set to null

        info.loc[info['DX_bl'] == 'EMCI', 'DX_bl'] = 'MCI'
        info.loc[info['DX_bl'] == 'LMCI', 'DX_bl'] = 'MCI'
        info.loc[info['DX_bl'] == 'SMC', 'DX_bl'] = 'CN'
        info.loc[info['DX'] == 'Dementia', 'DX'] = 'AD'
        # some blank DX in bl is filled with DX_bl
        info.loc[(info['VISCODE'] == 'bl')
                 & pd.isnull(info['DX']), 'DX'] = info.loc[(info['VISCODE'] == 'bl') & pd.isnull(info['DX']), 'DX_bl']

        info = info[['RID', 'VISCODE', 'DX', 'NEW_DX']]
        # info.to_csv('./data/info_merged_dx.csv', sep=',', header=True, index=True)
        # define sMCI, pMCI
        for i in info.index:
            if info.loc[i]['DX'] == 'CN':
                info.loc[i, 'NEW_DX'] = 'CN'
            elif info.loc[i]['DX'] == 'AD':
                info.loc[i, 'NEW_DX'] = 'AD'
            elif info.loc[i]['DX'] == 'MCI':
                rid = info.loc[i]['RID']
                dxs = info[info['RID'] == rid]['DX'].values
                if 'AD' in dxs:
                    info.loc[i, 'NEW_DX'] = 'pMCI'
                else:
                    info.loc[i, 'NEW_DX'] = 'sMCI'

        info = info[['RID', 'VISCODE', 'NEW_DX']]
        return info

    @staticmethod
    def _filtering(info: pd.DataFrame):
        '''filtering for AD->CN

        Args:
            info:

        Returns:

        '''
        p_rid = []
        _info = info.copy()[['ID', 'VISIT', 'DX']]
        _info['DX'] = _info['DX'].apply(
            func=lambda x: {'CN': 0, 'sMCI': 1, 'pMCI': 2, 'AD': 3, np.nan: -1}[x])

        _info['ID'] = _info['ID'].values.astype(int)
        _info = _info.sort_values(by=['ID', 'VISIT'], kind='mergesort')
        _info = _info[_info['DX'] != -1]
        for i in _info['ID'].drop_duplicates().values:
            temp = _info[_info['ID'] == i]
            if not (temp.sort_values(by=['VISIT'], kind='mergesort')['DX'].values ==
                    temp.sort_values(by=['DX'], kind='mergesort')['DX'].values).all():
                p_rid.append(i)
        ids = set(info['ID'].values) - set(p_rid)
        return ids

    @staticmethod
    def dx_mapping(dx, clfsetting):

        if clfsetting == 'CN-AD':
            dx_label = np.array(list(map(lambda x: {'CN': 0, 'sMCI': -1,
                                                    'pMCI': -1, 'MCI': -1, 'AD': 1, np.nan: -1}[x], dx)))
        elif clfsetting == 'CN_sMCI-pMCI_AD':
            dx_label = np.array(list(map(lambda x: {'CN': 0, 'sMCI': 0, 'pMCI': 1, 'AD': 1, np.nan: -1}[x], dx)))
        elif clfsetting == 'sMCI-pMCI':
            dx_label = np.array(list(map(lambda x: {'CN': -1, 'sMCI': 0, 'pMCI': 1, 'AD': -1, np.nan: -1}[x], dx)))
        elif clfsetting == 'DIS-NC':
            dx_label = np.array(list(map(lambda x: {'CN': 0, 'sMCI': -1, 'pMCI': -1, 'AD': 1, np.nan: -1}[x], dx)))
        elif clfsetting == 'regression':
            raise NotImplementedError
            # dx_label = np.array(list(map(lambda x: {'CN': 0, 'sMCI': 1, 'pMCI': 2, 'AD': 3, np.nan: 4}[x], dx)))
        else:
            raise NotImplementedError
        return dx_label

    def get_table(self, path, modals: Union[list, str], dsettings):

        cohort = dsettings['cohort']    #‘ALL‘
        label_names = dsettings['label_names']   #['DX']
        assert cohort in ['ADNI1', 'ADNI2', 'ADNI1-P', 'ADNI2-P', 'PET', 'NO-PET', 'ALL']

        info = self.get_info()
        FDG_INFO = self.get_fdg()
        AV45_INFO = self.get_av45()
        CSF_INFO = self.get_csf()
        AUX_INFO = self.get_aux()
        DX_INFO = self.get_dx()

        info['IMAGE'] = info[['RID', 'VISCODE']].apply(axis=1, func=lambda x: os.path.exists(
            os.path.join(path, x['RID'], 'report', 'catreport_' + x['VISCODE'] + '.pdf')))
        info.loc[info['IMAGE'], modals[0]] = info.loc[info['IMAGE'], ['RID', 'VISCODE']].apply(
            axis=1, func=lambda x: os.path.join(path, x['RID'], 'mri', modals[0] + x['VISCODE'] + '.nii'))

        for df in [DX_INFO, FDG_INFO, AV45_INFO, CSF_INFO, AUX_INFO]:
            info = info.merge(df, how='left', on=['RID', 'VISCODE'], )
        info['VISIT'] = info['VISCODE'].apply(lambda x: 0 if x == 'bl' else int(x.replace('m', '')))

        FDG = info[['Angular_Left', 'Angular_Right', 'CingulumPost_Bilateral',
                    'Temporal_Left', 'Temporal_Right', 'FDG_SUMMARY']].astype(float).values
        AV45 = info[['FRONTAL_SUVR', 'CINGULATE_SUVR', 'PARIETAL_SUVR',
                     'TEMPORAL_SUVR', 'COMPOSITE_REF_SUVR',
                     'SUMMARYSUVR_COMPOSITE_REFNORM', ]].astype(float).values

        # to garantee no overlapping
        adni1_subs = info[info['COLPROT'] == 'ADNI1']['RID'].drop_duplicates().values
        adni2_subs = info[info['COLPROT'] == 'ADNI2']['RID'].drop_duplicates().values
        pet_subs = info[~np.isnan(np.concatenate([FDG, AV45], axis=-1)).all(axis=1)]['RID'].drop_duplicates().values

        inx = ~pd.isnull(info).all(axis=1)
        if cohort in ['ADNI1', 'ADNI2']:
            if cohort == 'ADNI1':
                inx &= info['RID'].isin(adni1_subs)
            else:
                inx &= info['RID'].isin(adni2_subs) & (~info['RID'].isin(adni1_subs))
        elif cohort == 'PET':
            inx &= info['RID'].isin(pet_subs)
        elif cohort == 'NO-PET':
            inx &= ~info['RID'].isin(pet_subs)
        elif cohort == 'ALL':
            pass
        else:
            raise NotImplementedError

        info = info.loc[inx]
        info = info.rename(columns={'RID': 'ID', 'NEW_DX': 'DX', 'PTGENDER': 'GENDER'})

        return info

    @staticmethod
    def get_label(info, dsettings):
        '''return labels for training and groups of subjects
        '''
        cohort = dsettings['cohort']
        label_names = dsettings['label_names']

        AUX_LABEL = info[['AGE', 'GENDER', 'APOE4', 'mPACCdigit', 'ADASQ4', 'Hippocampus']].astype(float).values

        DX = info['DX_LABEL'].values
        FDG = info[['Angular_Left', 'Angular_Right', 'CingulumPost_Bilateral',
                    'Temporal_Left', 'Temporal_Right', 'FDG_SUMMARY']].astype(float).values
        AV45 = info[['FRONTAL_SUVR', 'CINGULATE_SUVR', 'PARIETAL_SUVR',
                     'TEMPORAL_SUVR', 'COMPOSITE_REF_SUVR',
                     'SUMMARYSUVR_COMPOSITE_REFNORM', ]].astype(float).values

        LABEL = []
        DX = DX.reshape(-1, 1)
        if 'DX' in label_names:
            LABEL.append(DX)
            assert len(label_names) == 1
        if 'FDG' in label_names:
            LABEL.append(FDG[:, -1:])
        if 'AV45' in label_names:
            LABEL.append(AV45[:, -1:])
        if 'FDG-ROI' in label_names:
            LABEL.append(FDG)
        if 'AV45-ROI' in label_names:
            LABEL.append(AV45)
        if 'ADASQ4' in label_names:
            LABEL.append(info['ADASQ4'].astype(float).values.reshape([-1, 1]))
        if 'mPACCdigit' in label_names:
            LABEL.append(info['mPACCdigit'].astype(float).values.reshape([-1, 1]))
        LABEL = np.array(list(zip(*LABEL)))
        return LABEL

class ABIDE(Dataset):
    def __init__(self, subset, clfsetting, modals, no_smooth=False, only_bl=False,
                 ids_exclude=(), ids_include=(), maskmat=None, seed=1234, preload=True, dsettings=None):
        image_path = ABIDE_T1PATH
        super(ABIDE, self).__init__(image_path, subset, clfsetting, modals, no_smooth=no_smooth,
                                    only_bl=only_bl, ids_exclude=ids_exclude, ids_include=ids_include,
                                    maskmat=maskmat, seed=seed, preload=preload, dsettings={})

    def get_table(self, path, modals: Union[list, str], dsettings):
        if not isinstance(modals, list):
            modals = [modals]

        info = pd.read_csv(os.path.join(path, 'ABIDE_7_15_2022.csv'), dtype=str)
        info = info.rename(columns={'Subject': 'ID', 'Group': 'DX', 'Age': 'AGE', 'Sex': 'GENDER','Visit':'VISIT'})
        #info['VISIT'] = 0

        for sub in info['ID']:
            for modal in modals[:1]:
                subj_path = os.path.join(path, 'ABIDE_T1_pred', sub)
                img_path = self.check_imgpath(subj_path=subj_path, modal=modal)
                if img_path is not None:
                    info.loc[info['ID'] == sub, modal] = img_path

        return info

    @staticmethod
    def _filtering(info: pd.DataFrame):
        subs = []
        for sub in info['ID']:
            if (info.loc[info['ID'] == sub, 'DX'] == 'Autism').all():
                subs.append(sub)
            elif (info.loc[info['ID'] == sub, 'DX'] == 'Control').all():
                subs.append(sub)
        return subs

    @staticmethod
    def dx_mapping(dx, clfsetting):
        if clfsetting == 'AUTISM-CONTROL':
            labels = map(lambda x: {'Control': 0, 'Autism': 1}[x], dx)
            labels = np.array(list(labels), dtype=int)

        else:
            raise NotImplementedError

        return labels


# class ABIDE(Dataset):
#     def __init__(self, subset, clfsetting, modals, no_smooth=False, only_bl=False,
#                  ids_exclude=(), ids_include=(), maskmat=None, seed=1234, preload=True, dsettings=None):
#         image_path = ABIDE_T1PATH #'/data/datasets/COBRE'
#         super(ABIDE, self).__init__(image_path, subset, clfsetting, modals, no_smooth=no_smooth,
#                                     only_bl=only_bl, ids_exclude=ids_exclude, ids_include=ids_include,
#                                     maskmat=maskmat, seed=seed, preload=preload, dsettings={})
#
#     def get_table(self, path, modals: Union[list, str], dsettings):
#         if not isinstance(modals, list):
#             modals = [modals]
#
#         info = pd.read_csv(os.path.join(path, 'ABIDE_7_15_2022.csv'), dtype=str)
#         info = info.rename(columns={'Subject': 'ID', 'Group': 'DX', 'Age': 'AGE', 'Sex': 'GENDER','Visit':'VISIT'})
#         #info['VISIT'] = 0
#
#
#         for modal in modals:
#             if modal in ['mw', 'mwp1', 'mwp2', 'mwp3']:
#                 for sub in info['ID']:
#                     subj_path = os.path.join(path, 'ABIDE_T1_pred', sub)
#                     img_path = self.check_imgpath(subj_path=subj_path, modal=modal)
#                     if img_path is not None:
#                         info.loc[info['ID'] == sub, modal] = img_path
#                         #print(modal, img_path)
#             elif 'fmri' in modal:
#                 snp_df = self.load_fmridata(modal = modal)
#                 # Brain_Hippocampus.eQTL_2_gene.tpm_top_25pct.txt
#                 info = info.merge(right=snp_df, how='left', on=['ID'])
#             else:
#                 raise NotImplementedError
#
#         info= info.dropna(axis=0, subset=modals)
#
#
#         return info
#
#     @staticmethod
#     def _filtering(info: pd.DataFrame):
#         subs = []
#         for sub in info['ID']:
#             if (info.loc[info['ID'] == sub, 'DX'] == 'Autism').all():
#                 subs.append(sub)
#             elif (info.loc[info['ID'] == sub, 'DX'] == 'Control').all():
#                 subs.append(sub)
#         return subs
#
#     @staticmethod
#     def dx_mapping(dx, clfsetting):
#         if clfsetting == 'AUTISM-CONTROL':
#             labels = map(lambda x: {'Control': 0, 'Autism': 1}[x], dx)
#             labels = np.array(list(labels), dtype=int)
#
#         elif clfsetting == 'DIS-NC':
#             labels = map(lambda x: {'Control': 0, 'Autism': 1}[x], dx)
#             labels = np.array(list(labels), dtype=int)
#
#         else:
#             raise NotImplementedError
#
#         return labels

class ATLAS_2(Dataset):
    def __init__(self, subset, clfsetting, modals, no_smooth=False, only_bl=False,
                 ids_exclude=(), ids_include=(), maskmat=None, seed=1234, preload=True, dsettings=None):
        image_path = ATLAS_2_PATH
        super(ATLAS_2, self).__init__(image_path, subset, clfsetting, modals, no_smooth=no_smooth,
                                    only_bl=only_bl, ids_exclude=ids_exclude, ids_include=ids_include,
                                    maskmat=maskmat, seed=seed, preload=preload, dsettings={})

    def get_table(self, path, modals: Union[list, str], dsettings):
        if not isinstance(modals, list):
            modals = [modals]

        info = pd.read_csv(os.path.join(path, 'ATLAS_2/ATLAS_2.0_AddControl_20221104_reduced.csv'),dtype=str)
        info = info.rename(columns={'Subject ID': 'ID'})
        info['VISIT'] = 0

        for sub in info['ID']:
            if (info.loc[info['ID'] == sub, 'DX'] == 'Stroke').all():
                for modal in modals[:1]:
                        subj_path = os.path.join(path, 'ATLAS_2_pred', 'Training', sub,'ses-1')
                        img_path = self.check_imgpath(subj_path=subj_path, modal=modal)
                        if img_path is not None:
                            info.loc[info['ID'] == sub, modal] = img_path
                        else:
                            subj_path = os.path.join(path, 'ATLAS_2_pred', 'Testing', sub,'ses-1')
                            img_path = self.check_imgpath(subj_path=subj_path, modal=modal)
                            if img_path is not None:
                                info.loc[info['ID'] == sub, modal] = img_path
            if (info.loc[info['ID'] == sub, 'DX'] == 'Control').all():
                for modal in modals[:1]:
                    subj_path = os.path.join(ABIDE_T1PATH, 'ABIDE_T1_pred', sub)
                    img_path = self.check_imgpath(subj_path=subj_path, modal=modal)
                    if img_path is not None:
                        info.loc[info['ID'] == sub, modal] = img_path
        return info

    @staticmethod
    def _filtering(info: pd.DataFrame):
        subs = []
        for sub in info['ID']:
            if (info.loc[info['ID'] == sub, 'DX'] == 'Stroke').all():
                subs.append(sub)
            elif (info.loc[info['ID'] == sub, 'DX'] == 'Control').all():
                subs.append(sub)
        return subs

    @staticmethod
    def dx_mapping(dx, clfsetting):
        if clfsetting == 'STROKE-CONTROL':
            labels = map(lambda x: {'Control': 0, 'Stroke': 1}[x], dx)
            labels = np.array(list(labels), dtype=int)

        else:
            raise NotImplementedError

        return labels

class MINDS(Dataset):
    def __init__(self, subset, clfsetting, modals, no_smooth=False, only_bl=False,
                 ids_exclude=(), ids_include=(), maskmat=None, seed=1234, preload=True, dsettings=None):
        image_path = MINDS_PATH
        super(MINDS, self).__init__(image_path, subset, clfsetting, modals, no_smooth=no_smooth,
                                    only_bl=only_bl, ids_exclude=ids_exclude, ids_include=ids_include,
                                    maskmat=maskmat, seed=seed, preload=preload, dsettings={})

    def get_table(self, path, modals: Union[list, str], dsettings):
        if not isinstance(modals, list):
            modals = [modals]

        info = pd.read_csv(os.path.join(path, 'MINDS/Dataset_list.csv'),dtype=str)
        info = info.rename(columns={'diagnosis': 'DX', 'old': 'AGE', 'male=1.female=2': 'GENDER'})
        info['VISIT'] = 0

        for sub in info['ID']:
            aux_group=info.loc[info['ID'] == sub,'aux_group'].values[0]#TODO：
            for modal in modals[:1]:
                subj_path = os.path.join(path, 'MINDS_pred', aux_group,sub)
                img_path = self.check_imgpath(subj_path=subj_path, modal=modal)
                if img_path is not None:
                    info.loc[info['ID'] == sub, modal] = img_path

        return info

    @staticmethod
    def _filtering(info: pd.DataFrame):
        subs = []
        for sub in info['ID']:
            if (info.loc[info['ID'] == sub, 'DX'] == 'Bipolar Disorder').all():
                subs.append(sub)
            elif (info.loc[info['ID'] == sub, 'DX'] == 'Healthy Control').all():
                subs.append(sub)
            elif (info.loc[info['ID'] == sub, 'DX'] == 'Major Depressive Disorder').all():
                subs.append(sub)
            elif (info.loc[info['ID'] == sub, 'DX'] == 'Schizophrenia').all():
                subs.append(sub)
        return subs

    @staticmethod
    def dx_mapping(dx, clfsetting):
        if clfsetting == 'STROKE-CONTROL':
            labels = map(lambda x: {'Healthy Control': 0, 'Bipolar Disorder': 1,'Major Depressive Disorder':1,'Schizophrenia':1}[x], dx)
            labels = np.array(list(labels), dtype=int)

        else:
            raise NotImplementedError

        return labels

class BraTS(Dataset):
    def __init__(self, subset, clfsetting, modals, no_smooth=False, only_bl=False,
                 ids_exclude=(), ids_include=(), maskmat=None, seed=1234, preload=True, dsettings=None):
        image_path = BraTS_PATH
        super(BraTS, self).__init__(image_path, subset, clfsetting, modals, no_smooth=no_smooth,
                                    only_bl=only_bl, ids_exclude=ids_exclude, ids_include=ids_include,
                                    maskmat=maskmat, seed=seed, preload=preload, dsettings={})

    def get_table(self, path, modals: Union[list, str], dsettings):
        if not isinstance(modals, list):
            modals = [modals]

        info = pd.read_csv(os.path.join(path, 'MICCAI_BraTS_20/BraTS_data.csv'), dtype=str)
        info['VISIT'] = 0

        for sub in info['ID']:
            if (info.loc[info['ID'] == sub, 'DX']!='Control').all():
                for modal in modals[:1]:
                    img_path = os.path.join(path, 'MICCAI_BraTS_20_pre', sub,'T1w.nii')
                    if os.path.exists(img_path):  #TODO:这样不便于把其他模态加进去，得改
                        info.loc[info['ID'] == sub, modal] = img_path
            if (info.loc[info['ID'] == sub, 'DX'] == 'Control').all():
                for modal in modals[:1]:
                    subj_path = os.path.join(ABIDE_T1PATH, 'ABIDE_T1_pred', sub)
                    img_path = self.check_imgpath(subj_path=subj_path, modal=modal)
                    if img_path is not None:
                        info.loc[info['ID'] == sub, modal] = img_path

        return info

    @staticmethod
    def _filtering(info: pd.DataFrame):
        subs = []
        for sub in info['ID']:
            if (info.loc[info['ID'] == sub, 'DX'] == 'HGG').all():
                subs.append(sub)
            elif (info.loc[info['ID'] == sub, 'DX'] == 'LGG').all():
                subs.append(sub)
            elif (info.loc[info['ID'] == sub, 'DX'] == 'unknown Tumor').all():
                subs.append(sub)
            elif (info.loc[info['ID'] == sub, 'DX'] == 'Control').all():
                subs.append(sub)
        return subs

    @staticmethod
    def dx_mapping(dx, clfsetting):
        if clfsetting == 'TUMOR-CONTROL':
            labels = map(lambda x: {'Control': 0, 'HGG': 1,'LGG':1,'unknown Tumor':1}[x], dx)
            labels = np.array(list(labels), dtype=int)

        else:
            raise NotImplementedError

        return labels

class NIFD(Dataset):

    def __init__(self, subset, clfsetting, modals, no_smooth=False, only_bl=False,
                 ids_exclude=(), ids_include=(), maskmat=None, seed=1234, preload=True, dsettings=None):
        image_path = NIFD_PATH
        super(NIFD, self).__init__(image_path, subset, clfsetting, modals, no_smooth=no_smooth,
                                    only_bl=only_bl, ids_exclude=ids_exclude, ids_include=ids_include,
                                    maskmat=maskmat, seed=seed, preload=preload, dsettings={})

    def get_table(self, path, modals: Union[list, str], dsettings):
        if not isinstance(modals, list):
            modals = [modals]

        info = pd.read_csv(os.path.join(path, 'NIFD_7_16_2022.csv'),dtype=str)
        info = info.rename(columns={'Subject': 'ID', 'Group': 'DX', 'Age': 'AGE', 'Sex': 'GENDER','Visit':'VISIT'})

        for sub in info['ID']:
            if (info.loc[info['ID'] == sub, 'DX']=='Patient').all():
                info.loc[info['ID'] == sub, 'DX'] ='FTD'
            for modal in modals[:1]:
                Acq_date_info_array = info.loc[(info['ID'] == sub), 'Acq Date'].values
                Acq_info_datetime={}
                for Acq_date_info in Acq_date_info_array:
                    month_info = Acq_date_info.split('/')[0]
                    day_info = Acq_date_info.split('/')[1]
                    year_info = Acq_date_info.split('/')[2]
                    if month_info[0]=='0':
                        month_info=month_info.split('0')[1]
                    if day_info[0]=='0':
                        day_info=day_info.split('0')[1]
                    Acq_info_datetime[Acq_date_info]=datetime.datetime(int(year_info), int(month_info), int(day_info))
                for Acq_date in sorted(os.listdir(os.path.join(path,'NIFD_T1_pre',sub))):
                    img_path = os.path.join(path, 'NIFD_T1_pre', sub,Acq_date,'t1_mprage.nii') #TODO:现在这种NIFD的处理方式是很粗糙的
                    date=Acq_date.split('_')[0]
                    year=date.split('-')[0]
                    month=date.split('-')[1]
                    day=date.split('-')[2]
                    if month[0] == '0':
                        month = month.split('0')[1]
                    if day[0] == '0':
                        day = day.split('0')[1]
                    Acq_datetime=datetime.datetime(int(year), int(month), int(day))
                    for k, v in Acq_info_datetime.items():
                        if v == Acq_datetime:
                            info.loc[(info['ID'] == sub) & (info['Acq Date']==k),modal] = img_path
                            break
                    break
        return info

    @staticmethod
    def _filtering(info: pd.DataFrame):
        subs = []
        for sub in info['ID']:
            if (info.loc[info['ID'] == sub, 'DX'] == 'FTD').all():
                subs.append(sub)
            elif (info.loc[info['ID'] == sub, 'DX'] == 'Control').all():
                subs.append(sub)
        return subs

    @staticmethod
    def dx_mapping(dx, clfsetting):
        if clfsetting == 'FTD-NC':
            labels = map(lambda x: {'Control': 0, 'FTD': 1}[x], dx)
            labels = np.array(list(labels), dtype=int)
        else:
            raise NotImplementedError

        return labels

class AIBL(Dataset):
    def __init__(self, subset, clfsetting, modals, no_smooth=False, only_bl=False,
                 ids_exclude=(), ids_include=(), maskmat=None, seed=1234, preload=True, dsettings=None):
        image_path = AIBL_PATH
        super(AIBL, self).__init__(image_path, subset, clfsetting, modals, no_smooth=no_smooth,
                                   only_bl=only_bl, ids_exclude=ids_exclude, ids_include=ids_include,
                                   maskmat=maskmat, seed=seed, preload=preload, dsettings={})

    def get_table(self, path, modals: Union[list, str], dsettings) -> pd.DataFrame:
        info = pd.read_csv(os.path.join(path, 'Data_extract_3.3.0/aibl_pdxconv_01-Jun-2018.csv'), dtype=str)

        # new_dx gen
        for i in info.index:
            if info.loc[i, 'DXCURREN'] == '1':
                info.loc[i, 'NEW_DX'] = 'CN'
            elif info.loc[i, 'DXCURREN'] == '3':
                info.loc[i, 'NEW_DX'] = 'AD'
            elif info.loc[i, 'DXCURREN'] == '2':
                rid = info.loc[i, 'RID']
                dxs = info[info['RID'] == rid]['DXCURREN'].values
                if '3' in dxs:
                    info.loc[i, 'NEW_DX'] = 'pMCI'
                else:
                    info.loc[i, 'NEW_DX'] = 'sMCI'
        info = info[~pd.isnull(info['NEW_DX'])]

        for i in info.index:
            rid = info.loc[i]['RID']
            vis = info.loc[i]['VISCODE']
            pdf_path = os.path.join(path, 'AIBL_pre', rid, 'report', 'catreport_' + vis + '.pdf')
            if os.path.exists(pdf_path):
                info.loc[i, modals[0]] = os.path.join(path, 'AIBL_pre', rid, 'mri', modals[0] + vis + '.nii')
        if len(modals) > 1:
            raise NotImplementedError

        info['VISIT'] = info['VISCODE'].apply(lambda x: 0 if x == 'bl' else int(x.replace('m', '')))
        info = info.rename(columns={'RID': 'ID', 'NEW_DX': 'DX', 'PTGENDER': 'GENDER'})

        return info

    @staticmethod
    def _filtering(info: pd.DataFrame):
        '''exclude for AD->CN
        '''
        p_rid = []
        _info = info.copy()[['ID', 'VISIT', 'DX']]
        _info['DX'] = _info['DX'].apply(
            func=lambda x: {'CN': 0, 'sMCI': 1, 'pMCI': 2, 'AD': 3, np.nan: -1}[x])

        _info['ID'] = _info['ID'].values.astype(int)
        _info = _info.sort_values(by=['ID', 'VISIT'], kind='mergesort')
        _info = _info[_info['DX'] != -1]
        for i in _info['ID'].drop_duplicates().values:
            temp = _info[_info['ID'] == i]
            if not (temp.sort_values(by=['VISIT'], kind='mergesort')['DX'].values ==
                    temp.sort_values(by=['DX'], kind='mergesort')['DX'].values).all():
                p_rid.append(i)
        ids = set(info['ID'].values) - set(p_rid)
        return ids

    @staticmethod
    def dx_mapping(dx, clfsetting):
        if clfsetting == 'CN-AD':
            dx_label = np.array(list(map(lambda x: {'CN': 0, 'sMCI': -1, 'pMCI': -1, 'AD': 1}[x], dx)))
        elif clfsetting == 'sMCI-pMCI':
            dx_label = np.array(list(map(lambda x: {'CN': -1, 'sMCI': 0, 'pMCI': 1, 'AD': -1}[x], dx)))
        elif clfsetting == 'DIS-NC':
            dx_label = np.array(list(map(lambda x: {'CN': 0, 'sMCI': -1, 'pMCI': -1, 'AD': 1}[x], dx)))
        else:
            raise NotImplementedError
        return dx_label


def get_dataset(dataset, clfsetting, modals, patch_size, batch_size, center_mat, flip_axises, no_smooth, no_shuffle,
                no_shift, n_threads, seed=1234, resample_patch=None, trtype='single', only_bl=False):

    if dataset == 'ADNI_dx':
        data_train = ADNI(subset=['training', 'testing'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                          seed=seed, only_bl=only_bl,
                          dsettings={'cohort': 'ADNI1', 'label_names': ['DX']})
        data_val = ADNI(subset=['validation'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                        seed=seed, only_bl=True,
                        dsettings={'cohort': 'ADNI1', 'label_names': ['DX']})
        data_test = ADNI(subset=['training', 'validation', 'testing'], clfsetting=clfsetting, modals=modals,
                         no_smooth=no_smooth, seed=seed, only_bl=True,
                         dsettings={'cohort': 'ADNI2', 'label_names': ['DX']})

    elif dataset == 'ADNI_DX':
        data_train = ADNI(subset=['training'], clfsetting=clfsetting, modals=modals,
                          no_smooth=no_smooth, seed=seed, only_bl=only_bl,
                          dsettings={'cohort': 'ALL', 'label_names': ['DX']})
        data_val = ADNI(subset=['validation'], clfsetting=clfsetting, modals=modals,
                        no_smooth=no_smooth, seed=seed, only_bl=True,
                        dsettings={'cohort': 'ALL', 'label_names': ['DX']})
        data_test = ADNI(subset=['testing'], clfsetting=clfsetting, modals=modals,
                         no_smooth=no_smooth, seed=seed, only_bl=True,
                         dsettings={'cohort': 'ALL', 'label_names': ['DX']})

    elif dataset == 'AIBL':
        data_train = AIBL(subset=['training'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                          seed=seed, only_bl=False)
        data_val = AIBL(subset=['validation'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                        seed=seed, only_bl=True)
        data_test = AIBL(subset=['testing'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                         seed=seed, only_bl=True)

    elif dataset=='ABIDE':
        data_train = ABIDE(subset=['training'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                           seed=seed, only_bl=True)
        data_val = ABIDE(subset=['validation'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                         seed=seed, only_bl=True)
        data_test = ABIDE(subset=['testing'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                          seed=seed, only_bl=True)

    elif dataset=='BraTS':
        data_train = BraTS(subset=['training'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                           seed=seed, only_bl=True)
        data_val = BraTS(subset=['validation'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                         seed=seed, only_bl=True)
        data_test = BraTS(subset=['testing'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                          seed=seed, only_bl=True)

    elif dataset=='ATLAS_2':
        data_train = ATLAS_2(subset=['training'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                           seed=seed, only_bl=True)
        data_val = ATLAS_2(subset=['validation'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                         seed=seed, only_bl=True)
        data_test = ATLAS_2(subset=['testing'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                          seed=seed, only_bl=True)

    elif dataset=='MINDS':
        data_train = MINDS(subset=['training'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                             seed=seed, only_bl=True)
        data_val = MINDS(subset=['validation'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                           seed=seed, only_bl=True)
        data_test = MINDS(subset=['testing'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                            seed=seed, only_bl=True)

    elif dataset == 'COMB':
        if clfsetting != 'DIS-NC':
            raise NotImplementedError
        atlas_train = ATLAS_2(subset=['training'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                           seed=seed, only_bl=True)
        atlas_val = ATLAS_2(subset=['validation'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                         seed=seed, only_bl=True)
        atlas_test = ATLAS_2(subset=['testing'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                          seed=seed, only_bl=True)

        brats_train = BraTS(subset=['training'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                           seed=seed, only_bl=True)
        brats_val = BraTS(subset=['validation'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                         seed=seed, only_bl=True)
        brats_test = BraTS(subset=['testing'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                          seed=seed, only_bl=True)

        minds_train = MINDS(subset=['training'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                             seed=seed, only_bl=True)
        minds_val = MINDS(subset=['validation'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                           seed=seed, only_bl=True)
        minds_test = MINDS(subset=['testing'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                            seed=seed, only_bl=True)

        abide_train = ABIDE(subset=['training'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                           seed=seed, only_bl=True)
        abide_val = ABIDE(subset=['validation'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                         seed=seed, only_bl=True)
        abide_test = ABIDE(subset=['testing'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                          seed=seed, only_bl=True)

        data_train = atlas_train + brats_train + minds_train + abide_train
        data_val = atlas_val + brats_val + minds_val + abide_val
        data_test = atlas_test + brats_test + minds_test + abide_test
    elif dataset == 'NIFD':
        data_train = NIFD(subset=['training'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                          seed=seed, only_bl=True)
        data_val = NIFD(subset=['validation'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                        seed=seed, only_bl=True)
        data_test = NIFD(subset=['testing'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                         seed=seed, only_bl=True)
    else:
        raise NotImplementedError

    if data_train:
        data_train = Patch_Data(data_train, patch_size=patch_size, center_mat=center_mat,
                                shift=not no_shift, flip_axis=flip_axises, resample_patch=resample_patch)
        data_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=not no_shuffle,
                                                 num_workers=n_threads, pin_memory=True)

    if data_val:
        data_val = Patch_Data(data_val, patch_size=patch_size, center_mat=center_mat,
                              shift=False, flip_axis=None, resample_patch=resample_patch)
        data_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False,
                                               num_workers=n_threads, pin_memory=True)

    data_test = Patch_Data(data_test, patch_size=patch_size, center_mat=center_mat,
                           shift=False, flip_axis=None, resample_patch=resample_patch)
    data_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False,
                                            num_workers=n_threads, pin_memory=True)

    if dataset not in ['AIBL']:
        checkpath = os.path.join(BASEDIR, 'utils', 'datacheck', dataset + '_' + clfsetting + '.pkl')
        if os.path.exists(checkpath):
            with open(checkpath, 'rb') as f:
                id_infos = pkl.load(f)
            # assert (np.sort(np.unique(id_infos[0])) == np.sort(np.unique(data_train.dataset.imgdata.id_info))).all()
            # assert (np.sort(np.unique(id_infos[1])) == np.sort(np.unique(data_val.dataset.imgdata.id_info))).all()
            # assert (np.sort(np.unique(id_infos[2])) == np.sort(np.unique(data_test.dataset.imgdata.id_info))).all()   #TODO：ATLAS2在这三行碰壁了
        else:
            if not os.path.exists(os.path.join(BASEDIR, 'utils', 'datacheck')):
                os.mkdir(os.path.join(BASEDIR, 'utils', 'datacheck'))
            with open(checkpath, 'wb') as f:
                pkl.dump([data_train.dataset.imgdata.id_info,
                          data_val.dataset.imgdata.id_info,
                          data_test.dataset.imgdata.id_info], f)

        a = set(data_train.dataset.imgdata.id_info)
        b = set(data_val.dataset.imgdata.id_info)
        c = set(data_test.dataset.imgdata.id_info)
        assert len(a.intersection(b)) == len(a.intersection(c)) == len(b.intersection(c)) == 0

    if trtype == 'single':
        dataloader_list = [[data_train, data_val, data_test]]
    elif trtype == '5-rep':
        dataloader_list = [[data_train, data_val, data_test] for i in range(5)]
    else:
        # TODO: trtype == k-fold
        raise NotImplementedError

    return dataloader_list, None, None
