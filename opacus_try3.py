import warnings
warnings.filterwarnings("ignore")
from utils.opts import parse_opts, mod_opt, BASEDIR, DATASEED, federated_opt, dp_opt
from utils.utils import occumpy_mem
from utils.datasets import get_dataset

# Step 1: Importing PyTorch and Opacus
import torch
from torchvision import datasets, transforms
import numpy as np
from opacus import PrivacyEngine

from opacus.dp_model_inspector import DPModelInspector
from opacus.utils import module_modification
from tqdm import tqdm
from utils.federated_datasets import get_dataset_federated
import warnings
warnings.filterwarnings("ignore")

# Step 2: Loading MNIST Data
# train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist', train=True, download=True,
#                                                           transform=transforms.Compose(
#                                                               [transforms.ToTensor(), transforms.Normalize((0.1307,),
#                                                                                                            (
#                                                                                                            0.3081,)), ]), ),
#                                            batch_size=64, shuffle=True, num_workers=1, pin_memory=True)
#
# test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist', train=False,
#                                                          transform=transforms.Compose(
#                                                              [transforms.ToTensor(), transforms.Normalize((0.1307,),
#                                                                                                           (
#                                                                                                           0.3081,)), ]), ),
#                                           batch_size=1024, shuffle=True, num_workers=1, pin_memory=True)

# Step 3: Creating a PyTorch Neural Network Classification Model and Optimizer
# model = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 8, 2, padding=3), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 1),
#                             torch.nn.Conv2d(16, 32, 4, 2), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 1),
#                             torch.nn.Flatten(),
#                             torch.nn.Linear(32 * 4 * 4, 32), torch.nn.ReLU(), torch.nn.Linear(32, 10))


from torch import nn
from einops import rearrange, repeat


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, num_patches, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.embed = nn.Linear(int(np.prod(patch_size) * channels), dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss()

    def forward(self, img):
        # x: batch, num_patch, channel, patch_size, patch_size, patch_size

        x = img.view(*img.shape[:2], -1)
        x = self.embed(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x),

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
                groundtruths.append(labels.numpy()[:, 0, :])  # multi patch
                group_labels.append(dis_label[:, 0])

        _probs = torch.stack([torch.cat([j[i] for j in predicts], dim=0) for i in range(len(predicts[1]))], dim=0)
        _probs = _probs.transpose(0, 1).cpu()

        predicts = np.array(
            [np.concatenate([j[i].cpu().numpy() for j in predicts], axis=0) for i in range(len(predicts[1]))])
        predicts = predicts.transpose((1, 0, 2))
        groundtruths = np.concatenate(groundtruths, axis=0)
        group_labels = np.concatenate([i.cpu().unsqueeze(-1).numpy() for i in group_labels], axis=0)


        groundtruths = groundtruths[:, :, -1:]
        predicts = predicts[:, :, -1:]

        non_nan = [torch.from_numpy(~np.isnan(groundtruths[:, i, :])) for i in range(groundtruths.shape[1])]
        val_loss = sum([self.criterion(_probs[:, i, :][non_nan[i]], torch.from_numpy(groundtruths[:, i, :])[non_nan[i]])
                        for i in range(groundtruths.shape[1])])

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
                groundtruths.append(labels.numpy()[:, 0, :])  # multi patch
                group_labels.append(dis_label[:,0])  #为啥会变成one-dimension？

        _probs = torch.stack([torch.cat([j[i] for j in predicts], dim=0) for i in range(len(predicts[1]))], dim=0)
        _probs = _probs.transpose(0, 1).cpu()

        predicts = np.array(
            [np.concatenate([j[i].cpu().numpy() for j in predicts], axis=0) for i in range(len(predicts[1]))])
        predicts = predicts.transpose((1, 0, 2))
        groundtruths = np.concatenate(groundtruths, axis=0)
        group_labels = np.concatenate([i.cpu().unsqueeze(-1).numpy() for i in group_labels], axis=0)


        groundtruths = groundtruths[:, :, -1:]  #这里维度变成2了?
        predicts = predicts[:, :, -1:]

        non_nan = [torch.from_numpy(~np.isnan(groundtruths[:, i, :])) for i in range(groundtruths.shape[1])]
        val_loss = sum([self.criterion(_probs[:, i, :][non_nan[i]], torch.from_numpy(groundtruths[:, i, :])[non_nan[i]])
                        for i in range(groundtruths.shape[1])])

        return predicts, groundtruths, group_labels, val_loss

    def fit(self, train_loader, optimizer, device, dtype):
        losses = torch.zeros(train_loader.dataset.labels.shape[1], dtype=dtype, device=device, )
        # train_loader={DataLoader:420}
        # train_Loader.dataset={Patch_Data:2096}
        self.train(True)
        for n, data in enumerate(train_loader, 0):
            inputs, aux_labels, labels, dis_label = data
            # multi patch
            labels = labels[:, 0, :]

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import numpy as np




def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

class SELayer(nn.Module):
    '''
    https://github.com/moskomule/senet.pytorch
    Modified
    '''

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
        return x * y.expand_as(x)

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

class SEBasicBlock(nn.Module):
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

        return out

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
                 sample_size=(121, 145, 121),
                 shortcut_type='B',
                 num_classes=400,
                 channels=(64, 128, 256, 512),
                 federated = None):
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
        self.fc = nn.Sequential(nn.Linear(fc_size * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
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
        x = self.fc(x)
        if self.num_classes == 1:
            x = torch.sigmoid(x)
        else:
            x = torch.log_softmax(x, dim=1)
        return x, #torch.zeros(x.shape, dtype=x.dtype, device=x.device)

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

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

# model = ResNet(BasicBlock, [1, 2, 2, 1], num_classes=1)
# try:
#     inspector = DPModelInspector()
#     inspector.validate(model)
#     print("Model's already Valid!\n")
# except:
#     model = module_modification.convert_batchnorm_modules(model)
#     inspector = DPModelInspector()
#     print(f"Is the model valid? {inspector.validate(model)}")
#     print("Model is convereted to be Valid!\n")

# optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# Step 4: Attaching a Differential Privacy Engine to the Optimizer
# privacy_engine = PrivacyEngine(model, batch_size=64, sample_size=60000, alphas=range(2, 32),
#                                noise_multiplier=1.3, max_grad_norm=1.0, )
# privacy_engine.attach(optimizer)


# Step 5: Training the private model over multiple epochs
def train(model, train_loader, optimizer, epoch, device, delta):
    model.train()
    loss = model.fit(train_loader,optimizer,device,dtype=torch.float32)
    epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
    print(
        f"Train Epoch: {epoch} \t"
        f"Loss: {loss.item():.6f} "
        f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")


if __name__ == "__main__":
    opt = parse_opts()
    opt.cuda_index = 3
    opt.method = 'Res18'

    if opt.no_cuda:
        device = 'cpu'
    else:
        device = torch.device("cuda:%d" % opt.cuda_index if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            occumpy_mem(device.index, percent=0.3)
            torch.cuda.set_device(device)
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

    # dataloader_list, train_user_groups, val_user_groups = get_dataset_federated(dataset='ABIDE',
    #                                                                   clfsetting='AUTISM-CONTROL', modals=opt.modals,
    #                                                                   patch_size=opt.patch_size,
    #                                                                   batch_size=opt.batch_size,
    #                                                                   center_mat=opt.center_mat,
    #                                                                   flip_axises=opt.flip_axises,
    #                                                                   no_smooth=opt.no_smooth,
    #                                                                   no_shuffle=opt.no_shuffle, no_shift=opt.no_shift,
    #                                                                   n_threads=opt.n_threads, seed=DATASEED,
    #                                                                   resample_patch=opt.resample_patch,
    #                                                                   trtype=opt.trtype
    #                                                                   )#federated_setting=opt.federated_setting
    dataloader_list, train_user_groups, val_user_groups = get_dataset(dataset='ABIDE',
                                                                      clfsetting='AUTISM-CONTROL', modals=opt.modals,
                                                                      patch_size=opt.patch_size,
                                                                      batch_size=opt.batch_size,
                                                                      center_mat=opt.center_mat,
                                                                      flip_axises=opt.flip_axises,
                                                                      no_smooth=opt.no_smooth,
                                                                      no_shuffle=opt.no_shuffle, no_shift=opt.no_shift,
                                                                      n_threads=opt.n_threads, seed=DATASEED,
                                                                      resample_patch=opt.resample_patch,
                                                                      trtype=opt.trtype
                                                                      )#federated_setting=opt.federated_setting

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

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    privacy_engine = PrivacyEngine(model, batch_size=64, sample_size=60000, alphas=range(2, 32),
                                   noise_multiplier=1.3, max_grad_norm=1.0, )
    privacy_engine.attach(optimizer)

    train_loader = dataloader_list[0][0]
    for epoch in range(1, 11):
        train(model, train_loader, optimizer, epoch, device="cuda:3", delta=1e-5)



