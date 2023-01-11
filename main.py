import sys
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import logging
import numpy as np
import torch
from utils.utils import occumpy_mem, mk_dirs, get_models
from utils.datasets import get_dataset
from utils.training import train, predict
from utils.opts import parse_opts, mod_opt, BASEDIR, DATASEED, federated_opt
from utils.landmarks import get_landmarks
from utils.parasearch import gen_paras

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

    ## generate landmarks
    if opt.gen_lmk:
        get_landmarks(opt.dataset, opt.clfsetting, opt.center_mat, opt.modals[0],
                      batch_size=opt.batch_size, patch_size=opt.patch_size, n_threads=opt.n_threads)
        sys.exit(0)

    ## generate datasets
    print('Training Dataset: ', opt.dataset)
    dataloader_list,train_user_groups, val_user_groups = get_dataset(dataset=opt.dataset,
                                  clfsetting=opt.clfsetting, modals=opt.modals,
                                  patch_size=opt.patch_size, batch_size=opt.batch_size,
                                  center_mat=opt.center_mat, flip_axises=opt.flip_axises,
                                  no_smooth=opt.no_smooth, no_shuffle=opt.no_shuffle, no_shift=opt.no_shift,
                                  n_threads=opt.n_threads, seed=DATASEED, resample_patch=opt.resample_patch,
                                  trtype=opt.trtype, federated_setting=opt.federated_setting)
                                #if trtype== 'single': [[data_train, data_val, data_test]]
                                #elif trtype == '5-rep':[[data_train, data_val, data_test] for i in range(5)]

    # get models
    pretrain_paths = [opt.pretrain_path for i in range(len(dataloader_list))]#default:None
    models = get_models(opt.method, opt.method_setting, opt.trtype, pretrain_paths=pretrain_paths, device=device,federated=opt.federated)

    if opt.para_search:
        # TODO: for the grid search of hyper-parameters
        best_metric = -np.inf
        best_models = None
        best_saved_path = None
        for paras in gen_paras(opt.method, opt.method_setting):
            models = get_models(opt.method, paras, '5-rep', pretrain_paths=pretrain_paths,
                                device=device)
            models, saved_paths = train(models, opt.n_epochs,
                                        dataloader_list,
                                        optimname=opt.optimizer,
                                        learning_rate=opt.learning_rate,
                                        device=device, logstring=TB_COMMENT,
                                        no_log=opt.no_log,
                                        no_val=opt.no_val,
                                        train_user_groups=train_user_groups,
                                        val_user_groups=val_user_groups,
                                        federated_setting=opt.federated_setting)

            reduced_result, metric_figlist, metric_strlist = predict(models, dataloader_list, device)
            cur_metric = float(reduced_result['AUC'].split('±'))
            if cur_metric > best_metric:
                best_metric = cur_metric
                best_models = models
                best_saved_path = saved_paths
        models = best_models
        saved_paths = best_saved_path
    elif not opt.no_train:
        ## training
        models, saved_paths = train(models, opt.n_epochs,
                                    dataloader_list,
                                    optimname=opt.optimizer,
                                    learning_rate=opt.learning_rate,
                                    device=device, logstring=TB_COMMENT,
                                    no_log=opt.no_log,
                                    no_val=opt.no_val,
                                    train_user_groups=train_user_groups,
                                    val_user_groups=val_user_groups,
                                    federated_setting=opt.federated_setting)

    ## prediction
    result = dict()
    if opt.pre_datasets is None:
        reduced_result, metric_figlist, metric_strlist = predict(models, dataloader_list, device)
        result[opt.dataset] = reduced_result
    else:
        for pre_dataset in opt.pre_datasets:
            dataloader_list,_,_ = get_dataset(dataset=pre_dataset, clfsetting=opt.clfsetting, modals=opt.modals,
                                          patch_size=opt.patch_size, batch_size=opt.batch_size,
                                          center_mat=opt.center_mat, flip_axises=opt.flip_axises,
                                          no_smooth=opt.no_smooth, no_shuffle=opt.no_shuffle, no_shift=opt.no_shift,
                                          n_threads=opt.n_threads, seed=DATASEED, resample_patch=opt.resample_patch,
                                          trtype=opt.trtype)
            reduced_result, metric_figlist, metric_strlist = predict(models, dataloader_list, device)
            result[pre_dataset] = reduced_result

    print('*****************************\r\n')
    print('Testing Result: \r\n %s\r\n' % result)
    print('*****************************')

    if (not opt.no_train) or opt.para_search:
        output_str = {'Method': opt.method,
                      'Dataset': opt.dataset,
                      'test_result': '%s' % result,
                      'saved_paths': '%s' % saved_paths,
                      'opt': '%s' % opt}
        print(output_str)
        fname = os.path.split(saved_paths[0])[-1].replace('.pth', '.txt')
        with open(os.path.join(BASEDIR, 'results', fname), 'a') as f:
            f.write(str(output_str))
