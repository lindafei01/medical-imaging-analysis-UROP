import warnings
warnings.filterwarnings("ignore")
from utils.opts import parse_opts, mod_opt, BASEDIR, DATASEED, federated_opt, dp_opt
from utils.utils import occumpy_mem
from utils.datasets import get_dataset
import copy
from opacus import PrivacyEngine
from opacus.dp_model_inspector import DPModelInspector
from opacus.utils import module_modification
from utils.federated_datasets import get_dataset_federated,DatasetSplit
from utils.training import predict
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import numpy as np
from models.resnet import *

# def conv3x3x3(in_planes, out_planes, stride=1):
#     # 3x3x3 convolution with padding
#     return nn.Conv3d(
#         in_planes,
#         out_planes,
#         kernel_size=3,
#         stride=stride,
#         padding=1,
#         bias=False)
#
# class SELayer(nn.Module):
#     '''
#     https://github.com/moskomule/senet.pytorch
#     Modified
#     '''
#
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1, 1)
#         return x * y.expand_as(x)
#
# def downsample_basic_block(x, planes, stride):
#     out = F.avg_pool3d(x, kernel_size=1, stride=stride)
#     zero_pads = torch.Tensor(
#         out.size(0), planes - out.size(1), out.size(2), out.size(3),
#         out.size(4)).zero_()
#     if isinstance(out.data, torch.cuda.FloatTensor):
#         zero_pads = zero_pads.cuda()
#
#     out = Variable(torch.cat([out.data, zero_pads], dim=1))
#
#     return out
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3x3(planes, planes)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
# class SEBasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(SEBasicBlock, self).__init__()
#         self.conv1 = conv3x3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3x3(planes, planes)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.downsample = downsample
#         self.stride = stride
#         self.se = SELayer(planes)
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out = self.se(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
# class Bottleneck(nn.Module):
#     # TODO: temperately use group conv
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm3d(planes)
#         # TODO(20201117): temperately use group conv
#         self.conv2 = nn.Conv3d(
#             planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=32)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm3d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
# class ResNet(nn.Module):
#     # modified by Zichao: for applying to MRI.
#     # 1. Same stride for 3 dim because of
#     # 2. input channel of conv1 to be 1
#     # 3. return extra output for compatibility reason
#     # 4. add sigmoid for fc
#     # 5. allow layer==0
#     # 6. allow modify channels
#     def __init__(self,
#                  block,
#                  layers,
#                  sample_size=(121, 145, 121),
#                  shortcut_type='B',
#                  num_classes=400,
#                  channels=(64, 128, 256, 512),
#                  federated = None):
#         self.inplanes = channels[0]
#         self.num_classes = num_classes
#         super(ResNet, self).__init__()
#         self.criterion = nn.BCELoss()
#         self.conv1 = nn.Conv3d(
#             1,
#             channels[0],
#             kernel_size=7,
#             stride=(2, 2, 2),
#             padding=(3, 3, 3),
#             bias=False)
#         self.bn1 = nn.BatchNorm3d(channels[0])
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
#         self.layer1 = self._make_layer(block, channels[0], layers[0], shortcut_type)
#         self.layer2 = self._make_layer(
#             block, channels[1], layers[1], shortcut_type, stride=2)
#         self.layer3 = self._make_layer(
#             block, channels[2], layers[2], shortcut_type, stride=2)
#         self.layer4 = self._make_layer(
#             block, channels[3], layers[3], shortcut_type, stride=2)
#
#         last_size = [int(math.ceil(i / 2 ** ((np.array(layers) > 0).sum() + 1))) for i in sample_size]
#         self.avgpool = nn.AvgPool3d(
#             (last_size[0], last_size[1], last_size[2]), stride=1)
#         # self.fc = nn.Linear(512 * block.expansion, num_classes)
#         fc_size = channels[np.where(np.array(layers) > 0)[0][-1]]
#         self.fc = nn.Sequential(nn.Linear(fc_size * block.expansion, num_classes))
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
#         if blocks < 1:
#             return nn.Sequential()
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             if shortcut_type == 'A':
#                 downsample = partial(
#                     downsample_basic_block,
#                     planes=planes * block.expansion,
#                     stride=stride)
#             else:
#                 downsample = nn.Sequential(
#                     nn.Conv3d(
#                         self.inplanes,
#                         planes * block.expansion,
#                         kernel_size=1,
#                         stride=stride,
#                         bias=False), nn.BatchNorm3d(planes * block.expansion))
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         if self.num_classes == 1:
#             x = torch.sigmoid(x)
#         else:
#             x = torch.log_softmax(x, dim=1)
#         return x, #torch.zeros(x.shape, dtype=x.dtype, device=x.device)
#
#     def embedding(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         return x.view(x.size(0), -1)
#
#     def fit(self, train_loader, optimizer, device, dtype):
#         losses = torch.zeros(train_loader.dataset.labels.shape[1], dtype=dtype, device=device, )
#         self.train(True)
#         for n, data in enumerate(train_loader, 0):
#             inputs, aux_labels, labels, dis_label = data
#
#             inputs = inputs.to(device=device, dtype=dtype)
#             labels = labels.to(device=device, dtype=dtype)
#             optimizer.zero_grad()
#             outputs = self(inputs)
#
#             for i in range(labels.shape[1]):
#                 assert outputs[i].shape == labels[:, i, :].shape
#                 non_nan = ~torch.isnan(labels[:, i, :])
#                 if non_nan.any():
#                     loss = self.criterion(outputs[i][non_nan], labels[:, i, :][non_nan])
#                     loss.backward(retain_graph=True)
#                     losses[i] += loss.detach()
#             optimizer.step()
#         return losses / len(train_loader)
#
#     def evaluate_data(self, val_loader, device, dtype='float32'):
#         predicts = []
#         groundtruths = []
#         group_labels = []
#
#         with torch.no_grad():
#             self.train(False)
#             for i, data in enumerate(val_loader, 0):
#                 inputs, aux_labels, labels, dis_label = data
#                 inputs = inputs.to(device=device, dtype=dtype)
#                 outputs = self(inputs)
#                 predicts.append(outputs)
#                 groundtruths.append(labels.numpy())
#                 group_labels.append(dis_label)
#
#         _probs = torch.stack([torch.cat([j[i] for j in predicts], dim=0) for i in range(len(predicts[1]))], dim=0)
#         _probs = _probs.transpose(0, 1).cpu()
#
#         predicts = np.array(
#             [np.concatenate([j[i].cpu().numpy() for j in predicts], axis=0) for i in range(len(predicts[1]))])
#         predicts = predicts.transpose((1, 0, 2))
#         groundtruths = np.concatenate(groundtruths, axis=0)
#         group_labels = np.concatenate([i.cpu().unsqueeze(-1).numpy() for i in group_labels], axis=0)
#
#         # for i, standr in enumerate(val_loader.dataset.standrs):
#         #     predicts[:, i, :] = standr.unstandr(predicts[:, i, :])
#         #     groundtruths[:, i, :] = standr.unstandr(groundtruths[:, i, :])
#
#         groundtruths = groundtruths[:, :, -1:]
#         predicts = predicts[:, :, -1:]
#
#         non_nan = [torch.from_numpy(~np.isnan(groundtruths[:, i, :])) for i in range(_probs.shape[1])]
#
#         val_loss = sum([self.criterion(_probs[:, i, :][non_nan[i]], torch.from_numpy(groundtruths[:, i, :])[non_nan[i]])
#                         for i in range(_probs.shape[1])])
#
#         return predicts, groundtruths, group_labels, val_loss
#
#     def predict_data(self, val_loader, device, dtype='float32'):
#         predicts = []
#         groundtruths = []
#         group_labels = []
#
#         with torch.no_grad():
#             self.train(False)
#             for i, data in enumerate(val_loader, 0):
#                 inputs, aux_labels, labels, dis_label = data
#                 inputs = inputs.to(device=device, dtype=dtype)
#                 outputs = self(inputs)
#                 predicts.append(outputs)
#                 groundtruths.append(labels.numpy())
#                 group_labels.append(dis_label)
#
#         _probs = torch.stack([torch.cat([j[i] for j in predicts], dim=0) for i in range(len(predicts[1]))], dim=0)
#         _probs = _probs.transpose(0, 1).cpu()
#
#         predicts = np.array(
#             [np.concatenate([j[i].cpu().numpy() for j in predicts], axis=0) for i in range(len(predicts[1]))])
#         predicts = predicts.transpose((1, 0, 2))
#         groundtruths = np.concatenate(groundtruths, axis=0)
#         group_labels = np.concatenate([i.cpu().unsqueeze(-1).numpy() for i in group_labels], axis=0)
#
#         # for i, standr in enumerate(val_loader.dataset.standrs):
#         #     predicts[:, i, :] = standr.unstandr(predicts[:, i, :])
#         #     groundtruths[:, i, :] = standr.unstandr(groundtruths[:, i, :])
#
#         groundtruths = groundtruths[:, :, -1:]
#         predicts = predicts[:, :, -1:]
#
#         non_nan = [torch.from_numpy(~np.isnan(groundtruths[:, i, :])) for i in range(_probs.shape[1])]
#
#         val_loss = sum([self.criterion(_probs[:, i, :][non_nan[i]], torch.from_numpy(groundtruths[:, i, :])[non_nan[i]])
#                         for i in range(_probs.shape[1])])
#
#         return predicts, groundtruths, group_labels, val_loss
#
# def get_fine_tuning_parameters(model, ft_begin_index):
#     if ft_begin_index == 0:
#         return model.parameters()
#
#     ft_module_names = []
#     for i in range(ft_begin_index, 5):
#         ft_module_names.append('layer{}'.format(i))
#     ft_module_names.append('fc')
#
#     parameters = []
#     for k, v in model.named_parameters():
#         for ft_module in ft_module_names:
#             if ft_module in k:
#                 parameters.append({'params': v})
#                 break
#         else:
#             parameters.append({'params': v, 'lr': 0.0})
#
#     return parameters
#
# def resnet18(**kwargs):
#     """Constructs a ResNet-18 model.
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     return model
#
# def resnet34(**kwargs):
#     """Constructs a ResNet-34 model.
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     return model
#
# def resnet50(**kwargs):
#     """Constructs a ResNet-50 model.
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     return model
#
# def resnet101(**kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     return model


def train(model, train_loader, optimizer, epoch, id, device, delta):
    model.train()
    loss = model.fit(train_loader,optimizer,device,dtype=torch.float32)
    epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
    print(
        f"Client {id}\t"
        f"Train Local Epoch: {epoch}\t"
        f"Loss: {loss.item():.6f}"
        f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")


class FLClient(nn.Module):
    def __init__(self, model, train_loader, optimname, device, delta, id, args):
        super(FLClient,self).__init__()
        self.model = model.to(device=device)
        self.train_loader = train_loader
        if optimname == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        else:
            raise NotImplementedError
        self.privacy_engine = PrivacyEngine(self.model, batch_size=64, sample_size=60000, alphas=range(2, 32),
                                       noise_multiplier=1.3, max_grad_norm=1.0, )
        self.privacy_engine.attach(self.optimizer)
        self.device = device
        self.delta = delta
        self.id = id
        self.args = args

    def update(self):
        local_epochs = self.args.federated_setting['local_epoch']
        for epoch in range(1, local_epochs+1):
            train(self.model, self.train_loader, self.optimizer, epoch, self.id, device=self.device, delta=self.delta)

    def recv(self,w):
        self.model.load_state_dict(copy.deepcopy(w))

#model,train_loader,optimizer,epoch,device,delta
class FLServer(nn.Module):
    def __init__(self, global_model, train_loaders:list, optimname:str, device, delta, args):
        super(FLServer, self).__init__()
        self.global_model = global_model
        self.num_client = args.federated_setting['num_users']
        self.clients = [FLClient(copy.deepcopy(global_model),train_loaders[i], optimname, device, delta, i, args)
                        for i in range(self.num_client)]
        self.args = args
        self.frac = self.args.federated_setting['frac']

    def aggregate(self,idxs_user):
        w = [self.clients[idx].model.state_dict() for idx in idxs_user]
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg

    def broadcast(self, new_par):
        for client in self.clients:
            client.recv(new_par)

    def global_update(self):
        idxs_users = np.random.choice(range(self.num_client), int(self.frac * self.num_client), replace=False)
        i = 0
        for idx in idxs_users:
            i += 1
            print(f"The Client {i} (idx = {idx}) is Training:\n*******************")
            self.clients[idx].update()
        self.broadcast(self.aggregate(idxs_users))
        torch.cuda.empty_cache()    #!!!!!!!!!!!!!!!!


if __name__ == "__main__":
    opt = parse_opts()
    opt.cuda_index = 1
    opt.global_epochs = 2
    opt.method = 'Res18'
    opt.dataset = 'ADNI_DX'#'BraTS'
    opt.clfsetting = 'CN-AD'#'TUMOR-CONTROL'
    opt.federated = True

    if opt.no_cuda:
        device = 'cpu'
    else:
        device = torch.device("cuda:%d" % opt.cuda_index if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            occumpy_mem(device.index, percent=0.3)
            torch.cuda.set_device(device)

    # device = 'cpu'
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


    model = resnet18(**opt.method_setting)
    try:
        inspector = DPModelInspector()
        inspector.validate(model)
        print("Model's already Valid!\n")
    except:
        model = module_modification.convert_batchnorm_modules(model)
        inspector = DPModelInspector()
        print(f"Is the model valid? {inspector.validate(model)}")
        print("Model is convereted to be Valid!\n")
    model = model.to(device=device)

    data_list, train_user_groups, val_user_groups = get_dataset_federated(dataset=opt.dataset,
                                                                          clfsetting=opt.clfsetting, modals=opt.modals,
                                                                          patch_size=opt.patch_size,
                                                                          batch_size=opt.batch_size,
                                                                          center_mat=opt.center_mat,
                                                                          flip_axises=opt.flip_axises,
                                                                          no_smooth=opt.no_smooth,
                                                                          no_shuffle=opt.no_shuffle,
                                                                          no_shift=opt.no_shift,
                                                                          n_threads=opt.n_threads, seed=DATASEED,
                                                                          resample_patch=opt.resample_patch,
                                                                          trtype=opt.trtype,
                                                                          federated_setting=opt.federated_setting)
    data_train = data_list[0][0]
    train_loaders = [None]*opt.federated_setting['num_users']
    for i in range(opt.federated_setting['num_users']):
        local_train_idxs = np.array([int(i) for i in train_user_groups[i]])
        train_loaders[i] = torch.utils.data.DataLoader(DatasetSplit(data_train, local_train_idxs),
                                    batch_size=opt.batch_size, shuffle=not opt.no_shuffle,
                                    num_workers=opt.n_threads, pin_memory=True)

    Server = FLServer(model, train_loaders, optimname='sgd', device=device, delta=1e-5, args=opt)
    for epoch in range(1, opt.global_epochs+1):
        Server.global_update()
        print(f'global epoch{epoch} has finished!')

    result = dict()
    dataloader_list, _, _ = get_dataset(dataset=opt.dataset, clfsetting=opt.clfsetting, modals=opt.modals,
                                        patch_size=opt.patch_size, batch_size=opt.batch_size,
                                        center_mat=opt.center_mat, flip_axises=opt.flip_axises,
                                        no_smooth=opt.no_smooth, no_shuffle=opt.no_shuffle, no_shift=opt.no_shift,
                                        n_threads=opt.n_threads, seed=DATASEED, resample_patch=opt.resample_patch,
                                        trtype=opt.trtype)
    reduced_result, metric_figlist, metric_strlist = predict([model], dataloader_list, device)
    result[opt.dataset] = reduced_result

    print('\r\n*****************************\r\n')
    print('Testing Result: \r\n %s\r\n' % result)
    print('*****************************')


