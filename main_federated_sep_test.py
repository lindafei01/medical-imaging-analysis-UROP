import sys
import os
import logging
import numpy as np
import torch
from utils.utils import occumpy_mem, mk_dirs, get_models
from utils.datasets import get_dataset
from utils.training import train, predict
from utils.opts import parse_opts, mod_opt, BASEDIR, DATASEED, federated_opt, dp_opt
from utils.landmarks import get_landmarks
from utils.parasearch import gen_paras
from utils.federated_training_sep_test import train_federated_sep
from utils.federated_datasets import get_dataset_federated

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level='WARNING')
if __name__ == "__main__":
    opt = parse_opts()
    logging.info("%s" % opt)
    if opt.no_cuda:
        device = 'cpu'
    else:
        device = torch.device("cuda:%d" % opt.cuda_index if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            occumpy_mem(device.index, percent=0.3)
            torch.cuda.set_device(device)
    mk_dirs(basedir=BASEDIR)
    opt.num_classes = 1
    TB_COMMENT = '_'.join([opt.method, '-'.join(opt.modals), opt.clfsetting, opt.log_label])
    opt = mod_opt(opt.method, opt)
    opt.method_setting.update(eval(opt.method_para))
    if opt.federated:
        opt = federated_opt(opt)
        if opt.federated_para:
            opt.federated_setting.update(eval(opt.federated_para))
    if opt.withDP:
        opt = dp_opt(opt)

    ## generate landmarks
    if opt.gen_lmk:
        get_landmarks(opt.dataset, opt.clfsetting, opt.center_mat, opt.modals[0],
                      batch_size=opt.batch_size, patch_size=opt.patch_size, n_threads=opt.n_threads)
        sys.exit(0)

    ## generate datasets
    print('Training Dataset: ', opt.dataset)

    data_list, train_user_groups, val_user_groups, test_user_groups = get_dataset_federated(dataset=opt.dataset,
                                                                      clfsetting=opt.clfsetting, modals=opt.modals,
                                                                      patch_size=opt.patch_size, batch_size=opt.batch_size,
                                                                      center_mat=opt.center_mat, flip_axises=opt.flip_axises,
                                                                      no_smooth=opt.no_smooth, no_shuffle=opt.no_shuffle,
                                                                      no_shift=opt.no_shift, n_threads=opt.n_threads, seed=DATASEED,
                                                                      resample_patch=opt.resample_patch, trtype=opt.trtype,
                                                                      federated_setting=opt.federated_setting)

    # get models
    pretrain_paths = [opt.pretrain_path for i in range(len(data_list))]

    origin_models = get_models(opt.method, opt.method_setting, opt.trtype, pretrain_paths=pretrain_paths, device=device, federated=opt.federated)


    #--------------------------------Training-----------------------------------------
    # global_model
    print("\r\n*************global model ready to be trained through federated learning*******************\r\n")
    global_models, saved_paths, dataloader_list = train_federated_sep(origin_models, opt.global_epochs,
                            data_list,
                            optimname=opt.optimizer,
                            learning_rate=opt.learning_rate,
                            device=device, logstring=TB_COMMENT,
                            no_log=opt.no_log,
                            no_val=opt.no_val,
                            train_user_groups=train_user_groups,
                            val_user_groups=val_user_groups,
                            test_user_groups=test_user_groups,
                            args=opt)
    print("\r\n*************global model has finished training through federated learning*******************\r\n")
        #得到global model了，dataloader_list里面每一个元素都是[None, None, test_loader]

    # local_model
    local_models = []
    for idx in range(opt.federated_setting['num_users']):
        print(f"\r\n*************local model {idx} ready to be trained*******************\r\n")
        models, saved_paths = train(origin_models, opt.global_epochs,
                                [dataloader_list[idx] for i in range(len(global_models))],
                                optimname=opt.optimizer,
                                learning_rate=opt.learning_rate,
                                device=device, logstring=TB_COMMENT,
                                no_log=opt.no_log,
                                no_val=opt.no_val,
                                args=opt)
        local_models.append(models)
        print(f"\r\n*************local model {idx} has finished training*******************\r\n")

    ## prediction
    local_results = []
    global_results = []
    for idx in range(opt.federated_setting['num_users']):
            assert opt.pre_datasets is None

            local_reduced_result, local_metric_figlist, local_metric_strlist = predict(local_models[idx],
                                                                     [dataloader_list[idx] for i in range(len(local_models[idx]))],
                                                                     device)
            global_reduced_result, global_metric_figlist, global_metric_strlist = predict(global_models,
                                                                     [dataloader_list[idx] for i in range(len(global_models))],
                                                                     device)
            local_results.append({opt.dataset:[idx, local_reduced_result]})
            global_results.append({opt.dataset:[idx, global_reduced_result]})



    print('*****************************\r\n')
    print('Local Test Result: \r\n %s\r\n' % local_results)
    print('Global Test Result: \r\n %s\r\n' % global_results)
    print('*****************************')

    if (not opt.no_train) or opt.para_search:
        output_str = {'Method': opt.method,
                      'Dataset': opt.dataset,
                      'local_test_result': '%s' % local_results,
                      'global_test_result': '%s' % global_results,
                      'saved_paths': '%s' % saved_paths,
                      'opt': '%s' % opt}
        print(output_str)
        fname = os.path.split(saved_paths[0])[-1].replace('.pth', '.txt')
        with open(os.path.join(BASEDIR, 'results', fname), 'a') as f:
            f.write(str(output_str))
