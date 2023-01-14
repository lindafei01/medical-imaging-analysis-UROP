from datetime import datetime
import copy
import torch
import socket
import os
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, recall_score, confusion_matrix
from sklearn.metrics import auc
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .opts import BASEDIR, DTYPE, LR_PATIENCE, LR_MUL_FACTOR, MOMENTUM, DAMPENING, WEIGHT_DECAY
from .utils import mf_save_model, mf_load_model, federated_logging
from .tools import ModifiedReduceLROnPlateau, Save_Checkpoint
from models.resnet import resnet18, resnet34
from models import DAMIDL, ViT
from models.vit import *
from torch.utils.data import DataLoader
from .federated_datasets import DatasetSplit
from opacus import PrivacyEngine
from .datasets import Patch_Data

method_map = {'Res18': resnet18, 'Res34': resnet34, 'DAMIDL': DAMIDL, 'ViT': ViT}

def train_epoch(epoch, global_step, train_loader, model, optimizer, writer, device):
    metric_strs = {}
    losses = model.fit(train_loader, optimizer, device, dtype=DTYPE)
    metric_strs['loss'] = losses.sum().item()
    if writer:
        for k, v in metric_strs.items():
            writer.add_scalar(k, v, global_step=global_step)
    print('local_epoch: [%d], loss_sum: %.2f' %
          (epoch + 1, losses.sum().item()))

def validate_epoch(global_step, val_loader, model, device, writer):
    def evaluation(model, val_loader):
        predicts, groundtruths, group_labels, val_loss = model.evaluate_data(val_loader, device, dtype=DTYPE)
        try:
            val_loss = val_loss.detach().cpu().item()
        except:
            pass
        predict1 = predicts[:, 0, :]
        groundtruth1 = groundtruths[:, 0, :]

        predict1 = np.array(predict1)
        groundtruth1 = np.array(groundtruth1)
        return predict1, groundtruth1, val_loss

    # monitor the performance for every subject
    pre, label, val_loss = evaluation(model=model, val_loader=val_loader)
    pre[np.isnan(pre)] = 0
    prec, rec, thr = precision_recall_curve(label, pre)
    fpr, tpr, thr = roc_curve(label, pre)
    test1 = confusion_matrix(y_pred=pre.round(), y_true=label).ravel()
    try:
        tn, fp, fn, tp = confusion_matrix(y_pred=pre.round(), y_true=label).ravel()
    except:
        tn, fp, fn, tp = confusion_matrix(y_pred=pre.round(), y_true=label, labels=[0,1]).ravel()

    metric_strs = {}
    metric_figs = {}
    metric_strs['AUC'] = auc(fpr, tpr)
    metric_strs['AUPR'] = auc(rec, prec)
    metric_strs['ACC'] = accuracy_score(y_pred=pre.round(), y_true=label)
    metric_strs['SEN'] = tp / (tp + fn)
    metric_strs['SPE'] = tn / (tn + fp)
    metric_strs['Val_Loss'] = val_loss

    if writer:
        for k, v in metric_strs.items():
            writer.add_scalar(k, v, global_step=global_step)

    val_metric = -metric_strs['Val_Loss']
    return metric_figs, metric_strs, val_metric

def predict_epoch(global_step, val_loader, model, device, writer):
    def evaluation(model, val_loader):
        predicts, groundtruths, group_labels, val_loss = model.predict_data(val_loader, device, dtype=DTYPE)
        try:
            val_loss = val_loss.detach().cpu().item()
        except:
            pass
        predict1 = predicts[:, 0, :]
        groundtruth1 = groundtruths[:, 0, :]

        predict1 = np.array(predict1)
        groundtruth1 = np.array(groundtruth1)
        return predict1, groundtruth1, val_loss

    # monitor the performance for every subject
    pre, label, val_loss = evaluation(model=model, val_loader=val_loader)
    pre[np.isnan(pre)] = 0
    prec, rec, thr = precision_recall_curve(label, pre)
    fpr, tpr, thr = roc_curve(label, pre)
    try:
        tn, fp, fn, tp = confusion_matrix(y_pred=pre.round(), y_true=label).ravel()
    except:
        tn, fp, fn, tp = confusion_matrix(y_pred=pre.round(), y_true=label, labels=[0, 1]).ravel()

    metric_strs = {}
    metric_figs = {}
    metric_strs['AUC'] = auc(fpr, tpr)
    metric_strs['AUPR'] = auc(rec, prec)
    metric_strs['ACC'] = accuracy_score(y_pred=pre.round(), y_true=label)
    metric_strs['SEN'] = tp / (tp + fn)
    metric_strs['SPE'] = tn / (tn + fp)
    metric_strs['Val_Loss'] = val_loss

    if writer:
        for k, v in metric_strs.items():
            writer.add_scalar(k, v, global_step=global_step)

    val_metric = -metric_strs['Val_Loss']
    return metric_figs, metric_strs, val_metric

def train_single(local_model, local_epochs, dataloaders, optimizer, learning_rate,
                 device, logstring, no_log, no_val):
    # the change for model is inplace
    train_loader, val_loader, test_loader = dataloaders
    local_model.to(device=device, dtype=DTYPE)

    if not no_log:
        writer = SummaryWriter(comment=logstring)
    else:
        writer = None

    scheduler = ModifiedReduceLROnPlateau(optimizer, 'min', patience=LR_PATIENCE, factor=LR_MUL_FACTOR, verbose=True)
    saved_path = os.path.join(BASEDIR, 'saved_model', datetime.now().strftime('%b%d_%H-%M-%S') + '_' +
                              socket.gethostname() + logstring + '.pth')
    save_checkpoint = Save_Checkpoint(save_func=mf_save_model, verbose=True,
                                      path=saved_path, trace_func=print, mode='min')

    for epoch in range(local_epochs):
        global_step = len(train_loader.dataset) * epoch
        #dataset是ADNI_DX:len(train_loader.dataset)=2096,batch_size=16,batch_sampler 131个，131*16=2096
        train_epoch(epoch, global_step, train_loader, local_model, optimizer, writer, device)
        if not no_val:
            metric_figs, metric_strs, val_metric = validate_epoch(
                global_step, val_loader, local_model, device, writer)
            print(metric_strs)
        else:
            val_metric = epoch

        scheduler.step(-val_metric)
        save_checkpoint(-val_metric, local_model)

    model = mf_load_model(model=local_model, path=saved_path, device=device)
    return model, saved_path

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def train_federated_single_sep(global_model, args, global_epochs, optimname, learning_rate,
                 device, logstring, no_log, no_val, dataloader_list):

    global_model.to(device=device, dtype=DTYPE)
    saved_path = os.path.join(BASEDIR, 'saved_model', datetime.now().strftime('%b%d_%H-%M-%S') + '_' +
                              socket.gethostname() + logstring + '.pth')

    for global_epoch in range(global_epochs): #global_epoch
        print(f'*****************************************\r\nglobal round {global_epoch+1} has started\r\n*****************************************')
        local_weights = []
        epsilons = np.zeros(args.federated_setting['num_users'])
        epsilon_log= []

        #挑选本全局轮次参与训练的用户
        m = max(int(args.federated_setting['frac'] * args.federated_setting['num_users']), 1)
        idxs_users = np.random.choice(range(args.federated_setting['num_users']), m, replace=False)

        for idx in idxs_users:
            local_train_loader = dataloader_list[idx][0]
            local_val_loader = dataloader_list[idx][1]
            local_test_loader = dataloader_list[idx][2]

            local_model = copy.deepcopy(global_model)

            if optimname == 'sgd':
                optimizer = optim.SGD(local_model.parameters(), lr=learning_rate, momentum=MOMENTUM,
                                      dampening=DAMPENING, weight_decay=WEIGHT_DECAY)
            else:
                optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)

            privacy_engine = None
            if args.withDP:
                privacy_engine = PrivacyEngine()
                local_model, optimizer, local_train_loader = privacy_engine.make_private_with_epsilon(
                    module=local_model,
                    optimizer=optimizer,
                    data_loader=local_train_loader,
                    epochs=args.federated_setting['local_epoch'],
                    target_epsilon=args.dp_setting['epsilon'],
                    target_delta=args.dp_setting['delta'],
                    max_grad_norm=args.dp_setting['max_grad_norm']
                )

            local_dataloaders = [local_train_loader, local_val_loader, local_test_loader]
            print(f'local client (idx:{idx}) is training...\r\n')
            local_model,_= train_single(local_model, args.federated_setting['local_epoch'], local_dataloaders, optimizer, learning_rate,
                 device, logstring, no_log, no_val)
            print(f'\r\nlocal client (idx:{idx}) finish!\r\n')

            if args.withDP:
                epsilons[idx] = privacy_engine.get_epsilon(args.dp_setting['delta'])

            w = local_model.state_dict()
            local_weights.append(copy.deepcopy(w))

        if args.withDP:
            epsilon_log.append(list(epsilons))
        else:
            epsilon_log = None
        print('parameter aggregation (FedAvg):')
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        mf_save_model(global_model,saved_path, )
        print(f'global round {global_epoch+1} finishes!')
        reduced_result, _, _ = predict([global_model], [[None,None,local_test_loader]], device)

        federated_logging(args, global_epoch, reduced_result, epsilon_log)

    global_model = mf_load_model(model=global_model, path=saved_path, device=device)
    return global_model,saved_path

def train_federated_sep(models, global_epochs, data_list: list, optimname, learning_rate,
          device, logstring, no_log, no_val,train_user_groups=None, val_user_groups=None, test_user_groups=None, args=None):
    trainedmodels = []
    saved_paths = []
    dataloader_list = []
    data_train, data_val, data_test = data_list[0]

    for idx in range(args.federated_setting['num_users']):
        local_train_idxs = np.array([int(i) for i in train_user_groups[idx]])
        local_val_idxs = np.array([int(i) for i in val_user_groups[idx]])
        local_test_idxs = np.array([int(i) for i in test_user_groups[idx]])

        local_data_train = Patch_Data(data_train, patch_size=args.patch_size, center_mat=args.center_mat,
                                      shift=not args.no_shift, flip_axis=args.flip_axises,
                                      resample_patch=args.resample_patch,
                                      idx=local_train_idxs)
        local_train_loader = torch.utils.data.DataLoader(local_data_train, batch_size=args.batch_size,
                                                         shuffle=not args.no_shuffle,
                                                         num_workers=args.n_threads, pin_memory=True)

        local_data_val = Patch_Data(data_val, patch_size=args.patch_size, center_mat=args.center_mat,
                                    shift=not args.no_shift, flip_axis=args.flip_axises,
                                    resample_patch=args.resample_patch,
                                    idx=local_val_idxs)
        local_val_loader = torch.utils.data.DataLoader(local_data_val, batch_size=args.batch_size,
                                                       shuffle=not args.no_shuffle,
                                                       num_workers=args.n_threads, pin_memory=True)

        local_data_test = Patch_Data(data_test, patch_size=args.patch_size, center_mat=args.center_mat,
                                    shift=not args.no_shift, flip_axis=args.flip_axises,
                                    resample_patch=args.resample_patch,
                                    idx=local_test_idxs)
        local_test_loader = torch.utils.data.DataLoader(local_data_test, batch_size=args.batch_size,
                                                       shuffle=not args.no_shuffle,
                                                       num_workers=args.n_threads, pin_memory=True)

        dataloader_list.append([local_train_loader, local_val_loader, local_test_loader])

    for i in range(len(data_list)):
        trained_global_model, saved_path = train_federated_single_sep(models[i], args, global_epochs, optimname,
                                                    learning_rate, device, logstring,
                                                    no_log, no_val, dataloader_list)
        trainedmodels.append(trained_global_model)
        saved_paths.append(saved_path)


    return trainedmodels, saved_paths, dataloader_list


def predict(models, dataloader_list: list, device):
    metric_figlist = []
    metric_strlist = []
    for i, model in enumerate(models):
        _,_,test_loader = dataloader_list[i]
        model = model.to(device)
        metric_figs, metric_strs, _ = predict_epoch(None, test_loader, model, device, writer=None)
        metric_figlist.append(metric_figs)
        metric_strlist.append(metric_strs)

    if len(models) == 1:
        reduced_result = {k: '%.3f' % v for k, v in metric_strlist[0].items()}
    else:
        values = np.array([[v for v in d.values()] for d in metric_strlist])
        keys = metric_strlist[0].keys()
        avg = values.mean(axis=0)
        std = values.std(axis=0)
        reduced_result = {k: '%.3f ± %.3f' % (avg, std) for k, avg, std in zip(keys, avg, std)}

    return reduced_result, metric_figlist, metric_strlist