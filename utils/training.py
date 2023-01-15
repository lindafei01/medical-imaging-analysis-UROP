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
from models import DAMIDL, ViT
from typing import Union
from torch.utils.data import DataLoader
from .federated_datasets import DatasetSplit
from opacus import PrivacyEngine
import csv
from quality_control.process_quality import *

method_map = {'Res18': resnet18, 'Res34': resnet34, 'DAMIDL': DAMIDL, 'ViT': ViT}

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
    try:
        tn, fp, fn, tp = confusion_matrix(y_pred=pre.round(), y_true=label).ravel()
    except:
        tn, fp, fn, tp = confusion_matrix(y_pred=pre.round(), y_true=label,labels=[0,1]).ravel()

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
    # y_pred = pre.round()
    # y_true = label
    # predict_quality = (y_pred == y_true)
    # log_path = '/data/home/feiyl/UROP/quality_control/quality_control.csv'
    # file = open(log_path, 'a+', encoding='utf-8', newline='')
    # csv_writer = csv.writer(file)
    # csv_writer.writerow(['img_path', 'prediction(True for truth, False for error)', 'IQR', 'rank'])
    # for i in range(len(val_loader.dataset)):
    #     img_path = val_loader.dataset.imgdata.namelist[i]
    #     process_quality = retrieve_COBRE_quality(img_path)
    #     csv_writer.writerow([val_loader.dataset.imgdata.namelist[i], predict_quality.squeeze()[i],
    #                          process_quality[0], process_quality[1]]) #TODO
    # file.close()

    prec, rec, thr = precision_recall_curve(label, pre)
    fpr, tpr, thr = roc_curve(label, pre)
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

def train_federated_single(model, args, n_epochs, dataloaders, optimname, learning_rate,
                 device, logstring, no_log, no_val,train_user_groups=None,val_user_groups=None,federated_setting=None):

    train_loader, val_loader, test_loader = dataloaders
    model.to(device=device, dtype=DTYPE)

    if optimname == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=MOMENTUM,
                              dampening=DAMPENING, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)



    saved_path = os.path.join(BASEDIR, 'saved_model', datetime.now().strftime('%b%d_%H-%M-%S') + '_' +
                              socket.gethostname() + logstring + '.pth')

    for epoch in range(n_epochs):
        local_weights = []
        local_models = []
        m = max(int(federated_setting['frac'] * federated_setting['num_users']), 1)
        idxs_users = np.random.choice(range(federated_setting['num_users']), m, replace=False)
        for idx in idxs_users:
            local_train_idxs = np.array([int(i) for i in train_user_groups[idx]])
            local_val_idxs=np.array([int(i) for i in val_user_groups[idx]])
            test1 = DatasetSplit(train_loader,local_train_idxs)
            local_train_loader = DataLoader(DatasetSplit(train_loader,local_train_idxs),
                                            shuffle=True)
            local_val_loader=DataLoader(DatasetSplit(val_loader,local_val_idxs))
            local_data=[local_train_loader,local_val_loader,test_loader]
            local_models[idx] = copy.deepcopy(model)
            privacy_engine = PrivacyEngine()
            local_model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=local_model,
                optimizer=optimizer,
                data_loader=local_train_loader,
                epochs=federated_setting['local_epoch'],
                target_epsilon=args.dp_setting['epsilon'],
                target_delta=args.dp_setting['delta'],
                max_grad_norm=args.dp_setting['max_grad_norm']
            )
            local_model,_= train_single(local_model, federated_setting['local_epoch'], local_data, optimname, learning_rate,
                 device, logstring, no_log, no_val)
            w = local_model.state_dict()
            local_weights.append(copy.deepcopy(w))
        global_weights = average_weights(local_weights)
        model.load_state_dict(global_weights)
        mf_save_model(model,saved_path, )

    model = mf_load_model(model=model, path=saved_path, device=device)
    return model,saved_path

def train(models, n_epochs, dataloader_list: list, optimname, learning_rate,
          device, logstring, no_log, no_val,train_user_groups=None, val_user_groups=None, federated_setting=None,args=None):
    # TODO: multiprocessing
    trainedmodels = []
    saved_paths = []

    for i, dataloaders in enumerate(dataloader_list):
        trainedmodel = None
        saved_path = None
        if not federated_setting:
            trainedmodel, saved_path = train_single(models[i], n_epochs, dataloaders, optimname,
                                                learning_rate, device, logstring,
                                                no_log, no_val)
        if federated_setting:
            trainedmodel, saved_path = train_federated_single(models[i], args, n_epochs, dataloaders, optimname,
                                                    learning_rate, device, logstring,
                                                    no_log, no_val, train_user_groups, val_user_groups)

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

    reduced_result = {}
    if len(models) == 1:
        for k, v in metric_strlist[0].items():
            if k!='SPE' and k!='SEN':
                reduced_result[k] = '%.3f' % v
        # reduced_result = {k: '%.3f' % v for k, v in metric_strlist[0].items()}
    else:
        values = np.array([[v for v in d.values()] for d in metric_strlist])
        keys = metric_strlist[0].keys()
        avg = values.mean(axis=0)
        std = values.std(axis=0)
        reduced_result = {k: '%.3f ± %.3f' % (avg, std) for k, avg, std in zip(keys, avg, std)}

    return reduced_result, metric_figlist, metric_strlist
