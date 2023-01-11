import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import numpy as np
import os


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


'''class SELayer(nn.Module):

    #https://github.com/moskomule/senet.pytorch
    #Modified

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)'''


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
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


'''class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = SELayer(planes)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out'''


class Bottleneck(nn.Module):
    # TODO: temperately use group conv
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        # TODO(20201117): temperately use group conv
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=32)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
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


class ResNet(nn.Module):
    # modified by Zichao: for applying to MRI.
    # 1. Same stride for 3 dim because of
    # 2. input channel of conv1 to be 1
    # 3. return extra output for compatibility reason
    # 4. add sigmoid for fc
    # 5. allow layer==0
    # 6. allow modify channels
    def __init__(self,
                 block,
                 layers,
                 dropout=0.2,
                 sample_size=(121, 145, 121),
                 aux_size=13366,  # (13366,8515,528)
                 hidden_size=(2048, 1024, 512),
                 shortcut_type='B',
                 num_classes=400,
                 channels=(64, 128, 256, 512)):
        self.inplanes = channels[0]
        self.num_classes = num_classes
        super(ResNet, self).__init__()
        self.criterion = nn.BCELoss()
        self.conv1 = nn.Conv3d(
            1,
            channels[0],
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, channels[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, channels[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, channels[3], layers[3], shortcut_type, stride=2)

        last_size = [int(math.ceil(i / 2 ** ((np.array(layers) > 0).sum() + 1))) for i in sample_size]
        self.avgpool = nn.AvgPool3d(
            (last_size[0], last_size[1], last_size[2]), stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        fc_size = channels[np.where(np.array(layers) > 0)[0][-1]]
        self.fc1 = nn.Sequential(nn.Linear(fc_size * block.expansion, fc_size * block.expansion), nn.Dropout(p=dropout))
        self.fc2 = nn.Sequential(nn.Linear(aux_size, hidden_size[0]), nn.Dropout(p=dropout),
                                 nn.Linear(hidden_size[0], hidden_size[1]),
                                 nn.Linear(hidden_size[1], hidden_size[2]), nn.Dropout(p=dropout))
        self.fc3 = nn.Sequential(nn.Linear(hidden_size[2] + fc_size * block.expansion, 256), nn.Dropout(p=dropout),
                                 nn.Linear(256, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        if blocks < 1:
            return nn.Sequential()
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, y):
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
        x = self.fc1(x)
        y = self.fc2((np.squeeze(y) + 1) / 2)
        x = self.fc3(torch.cat((x, y), dim=1))
        if self.num_classes == 1:
            x = torch.sigmoid(x)
        else:
            x = torch.log_softmax(x, dim=1)
        return x,  # torch.zeros(x.shape, dtype=x.dtype, device=x.device)

    def embedding(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def fit(self, train_loader, optimizer, device, dtype):
        losses = torch.zeros(train_loader.dataset.labels.shape[1], dtype=dtype, device=device, )
        self.train(True)
        for n, data in enumerate(train_loader, 0):
            inputs, aux_labels, labels, dis_labels = data

            inputs = inputs.to(device=device, dtype=dtype)
            aux_labels = aux_labels.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=dtype)
            optimizer.zero_grad()
            outputs = self(inputs, aux_labels)

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
                aux_labels = aux_labels.to(device=device, dtype=dtype)
                outputs = self(inputs, aux_labels)
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

        '''for i, standr in enumerate(val_loader.dataset.standrs):
            predicts[:, i, :] = standr.unstandr(predicts[:, i, :])
            groundtruths[:, i, :] = standr.unstandr(groundtruths[:, i, :])'''

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
        ft_module_names.append('layer{}'.format(i))
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


def resnet18_multimodal(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34_multimodal(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_multimodal(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101_multimodal(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


if __name__ == "__main__":
    i = torch.rand([16, 1, 121, 145, 121], device='cpu', dtype=torch.float32)
    d = torch.rand([16, 13366], device='cpu', dtype=torch.float32)
    net = ResNet(BasicBlock, [1, 2, 2, 1], num_classes=1)
    o = net(i, d)
    print(o)