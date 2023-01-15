from .datasets import *
import torch
import math

class DatasetSplit(torch.utils.data.DataLoader):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, patch_data, idxs, args, center_mat=None, shift=None, flip_axis=None):
        self.patch_data = patch_data
        self.idxs = [int(i) for i in idxs]
        self.batch_size = args.batch_size,
        self.shuffle = not args.no_shuffle,
        self.num_workers = args.n_threads,
        self.pin_memory = True
        super(DatasetSplit, self).__init__(torch.utils.data.DataLoader, idxs, center_mat, shift, flip_axis)


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.patch_data[self.idxs[item]]

def divisorGenerator(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor

def client_iid(data, num_users):
    num_items = int(len(data)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(data))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def client_iid_unequal(data, num_users):
    divisors = list(divisorGenerator(len(data)))
    if len(divisors) % 2 == 0:
        mid_idx = int((len(divisors) - 1) / 2)
    else:
        mid_idx = int(len(divisors) / 2 - 1)
    num_shards, num_imgs = int(divisors[mid_idx]), int(len(data) // divisors[mid_idx])

    min_shard = 0
    max_shard = 15
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    idx_shard = [i for i in range(num_shards)]

    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * (num_shards-num_users))
    random_shard_size = random_shard_size.astype(int) + 1

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # at least one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    return dict_users

def client_noniid_unequal(data, num_users):
    divisors = list(divisorGenerator(len(data)))
    if len(divisors) == 2:
        num_shards = int(math.sqrt(len(data)))
        num_imgs = int(len(data) // num_shards)
    else:
        if len(divisors) % 2 == 0:
            mid_idx = int((len(divisors) - 1) / 2)
        else:
            mid_idx = int(len(divisors) / 2 - 1)

        num_shards, num_imgs = int(divisors[mid_idx]), int(len(data) // divisors[mid_idx])

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = data.labels[:num_shards*num_imgs].squeeze()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 0
    max_shard = 15

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * (num_shards-num_users))
    random_shard_size = random_shard_size.astype(int) + 1

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # at least one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    return dict_users

def client_noniid(data, num_users):
    num_shards, num_imgs = num_users*2, int(len(data)//(num_users*2))
    idx_shard = [i for i in range(num_shards)]  #[0,1,2,...199]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = data.labels[:num_shards*num_imgs].squeeze()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

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

    elif dataset == 'COBRE':
        data_train = COBRE(subset=['training'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                           seed=seed, only_bl=True)
        data_val = COBRE(subset=['validation'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                         seed=seed, only_bl=True)
        data_test = COBRE(subset=['testing'], clfsetting=clfsetting, modals=modals, no_smooth=no_smooth,
                          seed=seed, only_bl=True)
    else:
        raise NotImplementedError

    if data_train:
        if federated_setting['iid']:
            if federated_setting['unequal']:
                train_user_groups = client_iid_unequal(data_train,federated_setting['num_users'])
            else:
                train_user_groups = client_iid(data_train, federated_setting['num_users'])
        else:
            # Sample Non-IID user data from Mnist
            if federated_setting['unequal']:
                train_user_groups = client_noniid_unequal(data_train, federated_setting['num_users'])
                # Chose uneuqal splits for every user
                # user_groups = client_noniid_unequal(data_train, args.num_users)
            else:
                train_user_groups = client_noniid(data_train, federated_setting['num_users'])
                # Chose euqal splits for every user
                # user_groups = client_noniid(data_train, args.num_users)
    else:
        train_user_groups = None

    if data_val:
        if federated_setting['iid']:
            if federated_setting['unequal']:
                val_user_groups = client_iid_unequal(data_val,federated_setting['num_users'])
            else:
                val_user_groups = client_iid(data_val, federated_setting['num_users'])
        else:
            # Sample Non-IID user data from Mnist
            if federated_setting['unequal']:
                val_user_groups = client_noniid_unequal(data_val,federated_setting['num_users'])
                # Chose uneuqal splits for every user
                # user_groups = client_noniid_unequal(data_train, args.num_users)
            else:
                val_user_groups = client_noniid(data_val,federated_setting['num_users'])
                # Chose euqal splits for every user
                # user_groups = client_noniid(data_train, args.num_users)
    else:
        val_user_groups = None

    if federated_setting['seperate_test_data']:
        if data_test:
            if federated_setting['iid']:
                if federated_setting['unequal']:
                    test_user_groups = client_iid_unequal(data_test, federated_setting['num_users'])
                else:
                    test_user_groups = client_iid(data_test, federated_setting['num_users'])
            else:

                if federated_setting['unequal']:
                    test_user_groups = client_noniid_unequal(data_test, federated_setting['num_users'])

                else:
                    test_user_groups = client_noniid(data_test, federated_setting['num_users'])

        else:
            test_user_groups = None
    else:
        test_user_groups = None


    if trtype == 'single':
        data_list = [[data_train, data_val, data_test]]
    elif trtype == '5-rep':
        data_list = [[data_train, data_val, data_test] for i in range(5)]
    else:
        # TODO: trtype == k-fold
        raise NotImplementedError

    return data_list, train_user_groups, val_user_groups, test_user_groups