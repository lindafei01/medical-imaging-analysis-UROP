from .datasets import *

class DatasetSplit(Patch_Data):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, patch_data, idxs, center_mat=None, shift=None, flip_axis=None):
        self.patch_data = patch_data
        self.idxs = [int(i) for i in idxs]
        super(DatasetSplit, self).__init__(patch_data, idxs, center_mat, shift, flip_axis)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.patch_data[self.idxs[item]]

def client_iid(data, num_users):
    num_items = int(len(data)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(data))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def client_noniid_unequal(data_train, num_users):
    raise NotImplementedError

def client_noniid(data_train, num_users):
    raise NotImplementedError


def get_dataset_federated(dataset, clfsetting, modals, patch_size, batch_size, center_mat, flip_axises, no_smooth, no_shuffle,
                no_shift, n_threads, seed=1234, resample_patch=None, trtype='single', only_bl=False,
                federated_setting=None):
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
    elif dataset == 'ABIDE':
        data_train = ABIDE(subset=['training'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                           seed=seed, only_bl=True)
        data_val = ABIDE(subset=['validation'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                         seed=seed, only_bl=True)
        data_test = ABIDE(subset=['testing'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                          seed=seed, only_bl=True)
    elif dataset == 'BraTS':
        data_train = BraTS(subset=['training'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                           seed=seed, only_bl=True)
        data_val = BraTS(subset=['validation'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                         seed=seed, only_bl=True)
        data_test = BraTS(subset=['testing'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                          seed=seed, only_bl=True)
    elif dataset == 'ATLAS_2':
        data_train = ATLAS_2(subset=['training'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                             seed=seed, only_bl=True)
        data_val = ATLAS_2(subset=['validation'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                           seed=seed, only_bl=True)
        data_test = ATLAS_2(subset=['testing'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                            seed=seed, only_bl=True)
    elif dataset == 'MINDS':
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
        # data_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=not no_shuffle,
        #                                          num_workers=n_threads, pin_memory=True)

    if data_val:
        data_val = Patch_Data(data_val, patch_size=patch_size, center_mat=center_mat,
                              shift=False, flip_axis=None, resample_patch=resample_patch)
        # data_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False,
        #                                      num_workers=n_threads, pin_memory=True)
    data_test = Patch_Data(data_test, patch_size=patch_size, center_mat=center_mat,
                           shift=False, flip_axis=None, resample_patch=resample_patch)
    # data_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False,
    #                                         num_workers=n_threads, pin_memory=True)

    if trtype == 'single':
        data_list = [[data_train, data_val, data_test]]
    elif trtype == '5-rep':
        data_list = [[data_train, data_val, data_test] for i in range(5)]
    else:
        # TODO: trtype == k-fold
        raise NotImplementedError

    if federated_setting['iid']:
        train_user_groups = client_iid(data_train, federated_setting['num_users'])
        val_user_groups = client_iid(data_val, federated_setting['num_users'])
    else:
        # Sample Non-IID user data from Mnist
        if federated_setting['unequal']:
            raise NotImplementedError
            # Chose uneuqal splits for every user
            # user_groups = client_noniid_unequal(data_train, args.num_users)
        else:
            raise NotImplementedError
            # Chose euqal splits for every user
            # user_groups = client_noniid(data_train, args.num_users)

    return data_list, train_user_groups, val_user_groups