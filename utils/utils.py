import numpy as np
import os
import torch
import logging
import time
from subprocess import Popen, PIPE
from models.resnet import resnet18, resnet34
from models import DAMIDL, ViT
from models.PreActivationResNet import PreActivationResNet18,PreActivationResNet34
from models.DenseNet import densenet121
from models.WideResnet import WideResNet18,WideResNet34
from models.Multimodal_ResNet import resnet18_multimodal
import datetime

method_map = {'Res18': resnet18, 'Res34': resnet34, 'DAMIDL': DAMIDL, 'ViT': ViT, 'densenet121':densenet121,'PreActivationResNet18':PreActivationResNet18,
              'PreActivationResNet34':PreActivationResNet34,'WideResNet18':WideResNet18,'WideResNet34':WideResNet34,'Res18_Multimodal':resnet18_multimodal}

def mk_dirs(basedir):
    dirs = [os.path.join(basedir, 'runs'),
            os.path.join(basedir, 'utils', 'datacheck'),
            os.path.join(basedir, 'saved_model'),
            os.path.join(basedir, 'results')]

    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

def get_models(method, method_setting, trtype: str, pretrain_paths: list, device, federated=None):
    models = []
    Model = method_map[method]
    if trtype == 'single':
        num_models = 1
    elif trtype == '5-rep':
        num_models = 5
    else:
        raise NotImplementedError

    for i in range(num_models):
        model = Model(**method_setting)
        if pretrain_paths[i] is not None:
            model = mf_load_model(model, pretrain_paths[i], device=device)
        models.append(model)
    return models

def mf_save_model(model, path, framework='pytorch'):
    if framework == 'pytorch':
        torch.save(model.state_dict(), path)
    elif framework == 'keras':
        model.save(path)
    else:
        raise NotImplementedError

def mf_load_model(model, path, framework='pytorch', device='cpu'):
    if framework == 'pytorch':
        try:
            model.load_state_dict(torch.load(path, map_location=device), strict=True)
        except RuntimeError:
            model.load_state_dict(torch.load(path, map_location=device), strict=False)
            logging.warning('Loaded pretrain train model Unstrictly!')
    elif framework == 'keras':
        model.load_weights(path)
    else:
        raise NotImplementedError
    return model

def check_mem(cuda_device):
    devices_info = os.popen(
        '"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split(
        "\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used

def occumpy_mem(cuda_device, percent=0.8):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * percent)
    block_mem = max_mem - used
    if block_mem < 0:
        return
    x = torch.FloatTensor(256, 1024, block_mem).cuda(cuda_device)
    del x

def count_params(model, framework='pytorch'):
    if framework == 'pytorch':
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
    elif framework == 'keras':
        params = model.count_params()
    else:
        raise NotImplementedError
    print('The network has {} params.'.format(params))

def parallel_cmd(cmds, num_p):
    procs = []
    from subprocess import check_output
    if num_p == 1:
        for cmd in cmds:
            print('Executing CMD: \r\n', cmd, '\r\n')
            yield check_output(cmd, shell=True), None
    else:
        for cmd in cmds:
            while True:
                dones = 0
                for p in procs:
                    code = p.poll()
                    if code is None:
                        pass
                    elif code == 0:
                        dones += 1
                    elif code != 0:
                        raise Exception(p.communicate()[1])
                if (len(procs) - dones) < num_p:
                    print('Executing CMD: \r\n', cmd, '\r\n')
                    # TODO: the preocess will stuck here when using keras.
                    procs.append(Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE))
                    time.sleep(10)  # incase duplicated saving path
                    break
                else:
                    time.sleep(10)
        for p in procs:
            p.wait()
            output, errinfo = p.communicate()
            if p.returncode != 0:
                raise Exception(errinfo.decode())
            else:
                yield output, errinfo

def federated_logging(args, global_epoch, test_result, epsilon_log=None):
    print(f"\nglobal epoch finished:{global_epoch + 1}\t logging...\n", )

    log_dir_test = '../federated_logs/test_log/'
    if not os.path.exists(log_dir_test):
        os.makedirs(log_dir_test)

    if args.withDP:
        assert epsilon_log
        exp_name = '_'.join([args.method, 'federated','DP'])
    else:
        assert epsilon_log == None
        exp_name = '_'.join([args.method, 'federated','w\oDP'])

    with open(log_dir_test + exp_name + '_test.txt', 'w') as f:
        f.write("%s\n" % test_result)

    if epsilon_log:
        log_dir_eps = '../federated_logs/privacy_log/'
        if not os.path.exists(log_dir_eps):
            os.makedirs(log_dir_eps)
        print("epsilons: max {:.2f},  mean {:.2f}, std {:.2f}\n".format(np.max(epsilon_log[-1]), np.mean(epsilon_log[-1]),
                                                                      np.std(epsilon_log[-1])))

        with open(log_dir_eps + exp_name + '_eps.txt', 'w') as f:
            for item in epsilon_log:
                f.write("%s\n" % item)
