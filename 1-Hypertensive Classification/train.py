import os
import sys
import random
import math
import subprocess
from glob import glob
from collections import OrderedDict
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm_notebook, tqdm
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision.datasets import ImageFolder
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import datasets, models, transforms
from torch.optim.optimizer import Optimizer, required
import torch.optim as optim

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

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

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        num_classes = 2
        self.label_linear = nn.Linear(512 * block.expansion, num_classes)
        self.relu_ = nn.ReLU(inplace=True)

        self.emb = nn.Embedding(num_classes, num_classes)
        self.emb.weight = nn.Parameter(torch.eye(num_classes))

    #         self.emb_mask = nn.Embedding(num_classes, num_classes)
    #         self.emb_mask.weight =  nn.Parameter(torch.Tensor([
    #             [1,1,0,0,0],
    #             [1,1,1,0,0],
    #             [0,1,1,1,0],
    #             [0,0,1,1,1],
    #             [0,0,0,1,1]]))
    #         self.emb_mask.weight.requires_grad_(False)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x2 = x
        x = self.last_linear(x)
        x2 = self.label_linear(x2.detach())
        return x, x2

    def forward(self, x, targets=None):
        x = self.features(x)
        x, x2 = self.logits(x)
        if targets is not None:
            tar = self.emb(targets).cuda()
        #             tar = tar * (self.emb_mask(targets).cuda())
        else:
            tar = None
        return x, x2, tar, self.emb.weight


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)

    pretrained_dict = model_zoo.load_url(settings['url'])
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    #     if pretrained is not None:
    #         settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
    #         initialize_pretrained_model(model, num_classes, settings)
    return model

def get_modified_model(model_name='se_resnext50_32x4d', num_outputs=None, pretrained=True,
                       freeze_bn=False, dropout_p=0, **kwargs):
    model = se_resnext50_32x4d()
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    in_features = model.last_linear.in_features

    if dropout_p == 0:
        model.last_linear = nn.Linear(in_features, num_outputs)
    else:
        model.last_linear = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_outputs))
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    return model


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def quadratic_weighted_kappa(y_pred, y_true):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.data.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.data.cpu().numpy()
    if y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0]
    else:
        y_pred = np.argmax(y_pred, axis=1)
    if(y_pred.all()==y_true.all()):
        return np.float64(1)
    return metrics.cohen_kappa_score(y_pred, y_true,weights='quadratic')


def comp_Loss(out1, out2, tar, emb_w, targets):
    """Compute the total loss"""
    out2_prob = F.softmax(out2, dim=1)
    tau2_prob = F.softmax(out2 / 2., dim=1).detach()
    soft_tar = F.softmax(tar, dim=1).detach()  # + 0.5*F.softmax(tar1).detach()
    #     soft_tar = (tar/torch.sum(tar, dim=1).unsqueeze(1)).detach()
    L_o1_y = F.cross_entropy(out1, targets)

    alpha = 0.9  # default:0.9
    beta = 0.5  # default:0.5#adjust_alpha(epoch)
    _, pred = torch.max(out2, 1)
    mask = pred.eq(targets).float().detach()
    L_o1_emb = -torch.mean(my_loss(out1, soft_tar))

    L_o2_y = F.cross_entropy(out2, targets)
    L_emb_o2 = -torch.sum(my_loss(tar, tau2_prob) * mask) / (torch.sum(mask) + 1e-8)
    gap = torch.gather(out2_prob, 1, targets.view(-1, 1)) - alpha
    L_re = torch.sum(F.relu(gap))
    # L2_loss = F.mse_loss(emb_w.t(), emb_w.detach())

    loss = beta * L_o1_y + (1 - beta) * L_o1_emb + L_o2_y + L_emb_o2 + L_re
    return loss


def my_loss(logit, prob):
    """ Cross-entropy function"""
    soft_logit = F.log_softmax(logit, dim=1)
    loss = torch.sum(prob * soft_logit, 1)
    return loss


def compute_accuracy(y_pred, y_true):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.data.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.data.cpu().numpy()
    if y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0]
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return metrics.accuracy_score(y_true, y_pred, normalize=True)


def my_train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    scores = AverageMeter()
    ac_scores = AverageMeter()
    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()

        output = model(input, target)
        out1, out2, tar, emb_w = output

        loss = comp_Loss(out1, out2, tar, emb_w, target)
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        score = quadratic_weighted_kappa(out1, target)
        ac_score = compute_accuracy(out1, target)

        losses.update(loss.item(), input.size(0))
        scores.update(score, input.size(0))
        ac_scores.update(ac_score, input.size(0))

    return losses.avg, scores.avg, ac_scores.avg


def my_validate(val_loader, model, criterion):
    losses = AverageMeter()
    scores = AverageMeter()
    ac_scores = AverageMeter()
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            output = model(input, target)
            out1, out2, tar, emb_w = output

            loss = comp_Loss(out1, out2, tar, emb_w, target)

            score = quadratic_weighted_kappa(out1, target)
            ac_score = compute_accuracy(out1, target)

            losses.update(loss.item(), input.size(0))
            scores.update(score, input.size(0))
            ac_scores.update(ac_score, input.size(0))
    print(losses.avg, scores.avg, ac_scores.avg)
    return losses.avg, scores.avg, ac_scores.avg





class CustomGaussianBlurWeighted(object):
    def __call__(self, pic):
        # 将PIL Image转换为NumPy数组
        img = np.array(pic)

        # 应用cv2.addWeighted操作
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 8), -4, 128)

        # 将处理后的NumPy数组转换回PIL Image
        return transforms.ToPILImage()(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CustomMedianBlurWeighted(object):
    def __call__(self, pic):
        img = np.array(pic)
        k = np.max(img.shape) // 20 * 2 + 1
        img = cv2.addWeighted(img, 4, cv2.medianBlur(img, k), -4, 128)
        return transforms.ToPILImage()(img)


def load_model_weights(model, model_path):
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()

    # 只保留存在于当前模型中的权重，并确保尺寸匹配
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}

    # 更新当前模型的权重
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)




if __name__ == '__main__':
    l1_probs = {}

    train_transform = []
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        CustomGaussianBlurWeighted(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        CustomGaussianBlurWeighted(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder('train', transform=train_transform)
    k_folds = 5  # 设定折数
    print('Training process initialize.')
    num_epochs = 10
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    model = get_modified_model(model_name='se_resnext50_32x4d', num_outputs=2, pretrained=False, freeze_bn=True,
                               dropout_p=0)

    load_model_weights(model, './model/se_resnext50_32x4d-a260b3a4.pth')
    model = model.cuda()  # 如果使用GPU

    print(next(model.parameters()).device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = 5
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # 分割数据集
        train_subset = Subset(dataset, train_ids)
        test_subset = Subset(dataset, test_ids)

        # 创建数据加载器
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4)
        print(len(train_loader))
        print(len(val_loader))
        # 在这里训练和验证模型
        for epoch in range(num_epochs):
            train_loss, train_score, train_ac_score = my_train(train_loader, model, criterion, optimizer, epoch)
            val_loss, val_score, val_ac_score = my_validate(val_loader, model, criterion)

            print(
                f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Score: {train_score:.4f}, Val Score: {val_score:.4f}')

            # 检查是否有最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                best_score = val_score
                torch.save(model.state_dict(), f'./new_model/model_weights_{fold + 1}.pth')
                print(f'Best Model for Fold {fold + 1} Saved')

        print(f'Best Loss for Fold {fold + 1}: {best_loss:.4f}, Best Score: {best_score:.4f}')

    print('Training process has finished.')
