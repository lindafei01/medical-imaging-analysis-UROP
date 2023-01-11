import logging
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
from .utils import mf_save_model, mf_load_model
from .tools import ModifiedReduceLROnPlateau, Save_Checkpoint
from models.resnet import resnet18, resnet34
from models import LDMIL, DAMIDL, ViT
from typing import Union
from torch.utils.data import DataLoader
from .datasets import DatasetSplit

method_map = {'Res18': resnet18, 'Res34': resnet34, 'LDMIL': LDMIL, 'DAMIDL': DAMIDL,
              'ViT': ViT}

def train_epoch(epoch, global_step, train_loader, model, optimizer, writer, device):
    metric_strs = {}
    losses = model.fit(train_loader, optimizer, device, dtype=DTYPE)
    metric_strs['loss'] = losses.sum().item()
    if writer:
        for k, v in metric_strs.items():
            writer.add_scalar(k, v, global_step=global_step)
    print('Epoch: [%d], loss_sum: %.2f' %
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
    tn, fp, fn, tp = confusion_matrix(y_pred=pre.round(), y_true=label).ravel()

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
    tn, fp, fn, tp = confusion_matrix(y_pred=pre.round(), y_true=label).ravel()

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

def train_single(model, n_epochs, dataloaders, optimname, learning_rate,
                 device, logstring, no_log, no_val):
    # the change for model is inplace
    train_loader, val_loader, test_loader = dataloaders
    model.to(device=device, dtype=DTYPE)

    if optimname == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=MOMENTUM,
                              dampening=DAMPENING, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if not no_log:
        writer = SummaryWriter(comment=logstring)
    else:
        writer = None

    scheduler = ModifiedReduceLROnPlateau(optimizer, 'min', patience=LR_PATIENCE, factor=LR_MUL_FACTOR, verbose=True)
    saved_path = os.path.join(BASEDIR, 'saved_model', datetime.now().strftime('%b%d_%H-%M-%S') + '_' +
                              socket.gethostname() + logstring + '.pth')
    save_checkpoint = Save_Checkpoint(save_func=mf_save_model, verbose=True,
                                      path=saved_path, trace_func=print, mode='min')

    for epoch in range(n_epochs):
        global_step = len(train_loader.dataset) * epoch
        #dataset是ADNI_DX:len(train_loader.dataset)=2096,batch_size=16,batch_sampler 131个，131*16=2096
        train_epoch(epoch, global_step, train_loader, model, optimizer, writer, device)
        if not no_val:
            metric_figs, metric_strs, val_metric = validate_epoch(
                global_step, val_loader, model, device, writer)
            print(metric_strs)
        else:
            val_metric = epoch

        scheduler.step(-val_metric)
        save_checkpoint(-val_metric, model)

    model = mf_load_model(model=model, path=saved_path, device=device)
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

def train_federated_single(model, n_epochs, dataloaders, optimname, learning_rate,
                 device, logstring, no_log, no_val,train_user_groups=None,val_user_groups=None,federated_setting=None):
    train_loader, val_loader, test_loader = dataloaders
    model.to(device=device, dtype=DTYPE)

    if optimname == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=MOMENTUM,
                              dampening=DAMPENING, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if not no_log:
        writer = SummaryWriter(comment=logstring)
    else:
        writer = None

    scheduler = ModifiedReduceLROnPlateau(optimizer, 'min', patience=LR_PATIENCE, factor=LR_MUL_FACTOR, verbose=True)
    saved_path = os.path.join(BASEDIR, 'saved_model', datetime.now().strftime('%b%d_%H-%M-%S') + '_' +
                              socket.gethostname() + logstring + '.pth')
    save_checkpoint = Save_Checkpoint(save_func=mf_save_model, verbose=True,
                                      path=saved_path, trace_func=print, mode='min')
    for epoch in range(n_epochs):
        local_weights=[]
        local_losses=[]
        ## 选定本轮参与训练的client
        m = max(int(federated_setting['frac'] * federated_setting['num_users']), 1)  # 每一轮次选择m个user参与训练
        idxs_users = np.random.choice(range(federated_setting['num_users']), m, replace=False)
        for idx in idxs_users:
            local_train_idxs = np.array([int(i) for i in train_user_groups[idx]])
            local_val_idxs=np.array([int(i) for i in val_user_groups[idx]])
            local_train_loader = DataLoader(DatasetSplit(train_loader,local_train_idxs),
                                            shuffle=True)   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!这个可能有很大问题
            local_val_loader=DataLoader(DatasetSplit(val_loader,local_val_idxs))           #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!这个可能有很大问题
            local_data=[local_train_loader,local_val_loader,test_loader]
            local_model = copy.deepcopy(model)
            local_model,_=train_single(local_model, federated_setting['local_epoch'], local_data, optimname, learning_rate,
                 device, logstring, no_log, no_val)
            w = local_model.state_dict()
            local_weights.append(copy.deepcopy(w))
        global_weights = average_weights(local_weights)
        model.load_state_dict(global_weights)

    model = mf_load_model(model=model, path=saved_path, device=device)
    return model,saved_path

def train(models, n_epochs, dataloader_list: list, optimname, learning_rate,
          device, logstring, no_log, no_val,train_user_groups=None,val_user_groups=None,federated_setting=None):
    # TODO: multiprocessing
    trainedmodels = []
    saved_paths = []

    for i, dataloaders in enumerate(dataloader_list):
        if not federated_setting:
            trainedmodel, saved_path = train_single(models[i], n_epochs, dataloaders, optimname,
                                                learning_rate, device, logstring,
                                                no_log, no_val)
        if federated_setting:
            trainedmodel, saved_path = train_federated_single(models[i], n_epochs, dataloaders, optimname,
                                                    learning_rate, device, logstring,
                                                    no_log, no_val, train_user_groups, val_user_groups,
                                                    federated_setting)

        trainedmodels.append(trainedmodel)
        saved_paths.append(saved_path)

    return trainedmodels, saved_paths


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
