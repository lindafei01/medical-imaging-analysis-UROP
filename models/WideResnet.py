import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import numpy as np

#__all__ = ['WideResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']
__all__ = ['WideResNet',  'WideResNet18','WideResNet34','WideResNet50']

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class WideBasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(WideBasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class WideBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(WideBottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class WideResNet(nn.Module):

    def __init__(self, block, layers, sample_size1, sample_size2, sample_duration, k=2, shortcut_type='B', num_classes=400, last_fc=True):
        self.last_fc = last_fc

        self.inplanes = 64
        super(WideResNet, self).__init__()
        self.num_classes=num_classes
        self.criterion=nn.BCELoss()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 * k, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128 * k, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256 * k, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512 * k, layers[3], shortcut_type, stride=2)
        last_duration = math.ceil(sample_duration / 16)
        last_size1 = math.ceil(sample_size1 / 32)
        last_size2 = math.ceil(sample_size2/32)
        self.avgpool = nn.AvgPool3d((last_duration, last_size1, last_size2), stride=1)
        self.fc = nn.Linear(512 * k * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.last_fc:
            x = self.fc(x)
        if self.num_classes == 1:
            x = torch.sigmoid(x)
        else:
            x = torch.log_softmax(x, dim=1)
        return x,

    def fit(self, train_loader, optimizer, device, dtype):
        losses = torch.zeros(train_loader.dataset.labels.shape[1], dtype=dtype, device=device, )
        self.train(True)
        for n, data in enumerate(train_loader, 0):
            inputs, aux_labels, labels, dis_label = data

            inputs = inputs.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=dtype)
            optimizer.zero_grad()
            outputs = self(inputs)

            for i in range(labels.shape[1]):
                assert outputs[i].shape == labels[:, i, :].shape
                non_nan = ~torch.isnan(labels[:, i, :])
                if non_nan.any():
                    loss = self.criterion(outputs[i][non_nan], labels[:, i, :][non_nan])
                    loss.backward(retain_graph=True)
                    losses[i] += loss.detach()
            optimizer.step()
        return losses / len(train_loader)

    def evaluate_data(self, val_loader, device, dtype='float32'):
        predicts = []
        groundtruths = []
        group_labels = []

        with torch.no_grad():
            self.train(False)
            for i, data in enumerate(val_loader, 0):
                inputs, aux_labels, labels, dis_label = data
                inputs = inputs.to(device=device, dtype=dtype)
                outputs = self(inputs)
                predicts.append(outputs)
                groundtruths.append(labels.numpy())
                group_labels.append(dis_label)

        _probs = torch.stack([torch.cat([j[i] for j in predicts], dim=0) for i in range(len(predicts[1]))], dim=0)
        _probs = _probs.transpose(0, 1).cpu()

        predicts = np.array(
            [np.concatenate([j[i].cpu().numpy() for j in predicts], axis=0) for i in range(len(predicts[1]))])
        predicts = predicts.transpose((1, 0, 2))
        groundtruths = np.concatenate(groundtruths, axis=0)
        group_labels = np.concatenate([i.cpu().unsqueeze(-1).numpy() for i in group_labels], axis=0)

        # for i, standr in enumerate(val_loader.dataset.standrs):
        #     predicts[:, i, :] = standr.unstandr(predicts[:, i, :])
        #     groundtruths[:, i, :] = standr.unstandr(groundtruths[:, i, :])

        groundtruths = groundtruths[:, :, -1:]
        predicts = predicts[:, :, -1:]

        non_nan = [torch.from_numpy(~np.isnan(groundtruths[:, i, :])) for i in range(_probs.shape[1])]

        val_loss = sum([self.criterion(_probs[:, i, :][non_nan[i]], torch.from_numpy(groundtruths[:, i, :])[non_nan[i]])
                        for i in range(_probs.shape[1])])

        return predicts, groundtruths, group_labels, val_loss

    def predict_data(self, val_loader, device, dtype='float32'):
        predicts = []
        groundtruths = []
        group_labels = []

        with torch.no_grad():
            self.train(False)
            for i, data in enumerate(val_loader, 0):
                inputs, aux_labels, labels, dis_label = data
                inputs = inputs.to(device=device, dtype=dtype)
                outputs = self(inputs)
                predicts.append(outputs)
                groundtruths.append(labels.numpy())
                group_labels.append(dis_label)

        _probs = torch.stack([torch.cat([j[i] for j in predicts], dim=0) for i in range(len(predicts[1]))], dim=0)
        _probs = _probs.transpose(0, 1).cpu()

        predicts = np.array(
            [np.concatenate([j[i].cpu().numpy() for j in predicts], axis=0) for i in range(len(predicts[1]))])
        predicts = predicts.transpose((1, 0, 2))
        groundtruths = np.concatenate(groundtruths, axis=0)
        group_labels = np.concatenate([i.cpu().unsqueeze(-1).numpy() for i in group_labels], axis=0)

        # for i, standr in enumerate(val_loader.dataset.standrs):
        #     predicts[:, i, :] = standr.unstandr(predicts[:, i, :])
        #     groundtruths[:, i, :] = standr.unstandr(groundtruths[:, i, :])

        groundtruths = groundtruths[:, :, -1:]
        predicts = predicts[:, :, -1:]

        non_nan = [torch.from_numpy(~np.isnan(groundtruths[:, i, :])) for i in range(_probs.shape[1])]

        val_loss = sum([self.criterion(_probs[:, i, :][non_nan[i]], torch.from_numpy(groundtruths[:, i, :])[non_nan[i]])
                        for i in range(_probs.shape[1])])

        return predicts, groundtruths, group_labels, val_loss

def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(ft_begin_index))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters

def WideResNet18(**kwargs):
    model=WideResNet(WideBasicBlock,[2,2,2,2],**kwargs)
    return model

def WideResNet34(**kwargs):
    model=WideResNet(WideBasicBlock,[3,4,6,3],**kwargs)
    return model

def WideResNet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = WideResNet(WideBottleneck, [3, 4, 6, 3], **kwargs)
    return model

if __name__ == '__main__':
    def count_params(model, framework='pytorch'):
        if framework == 'pytorch':
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
        elif framework == 'keras':
            params = model.count_params()
        else:
            raise NotImplementedError
        print('The network has {} params.'.format(params))
    net=WideResNet(WideBasicBlock,[2,2,2,2],sample_size1=117,sample_size2=141,sample_duration=117,num_classes=1)
    count_params(net)

