#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :EfficientNet.py
# @Time     :2021/3/29 下午2:14
# @Author   :Chang Qing


import torch
import torchvision
import torch.nn.functional as F

from torch import nn
from PIL import Image
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.model import MBConvBlock
from efficientnet_pytorch.utils import Swish, MemoryEfficientSwish
from torchvision import transforms


class Efficietnet_b4(nn.Module):
    def __init__(self, num_classes):
        super(Efficietnet_b4, self).__init__()
        self.basemodel1 = EfficientNet.from_pretrained('efficientnet-b4')

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(1792, num_classes)

    def forward(self, inputs):
        bs = inputs.size(0)
        x = self.basemodel1.extract_features(inputs)
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        # x = self._dropout(x)
        x = self._fc(x)
        return x


class Efficietnet_b5(nn.Module):

    def __init__(self, num_classes):
        super(Efficietnet_b5, self).__init__()
        self.basemodel1 = EfficientNet.from_name('efficientnet-b5')
        # self.basemodel1 = EfficientNet.from_pretrained('efficientnet-b5')
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(2048, num_classes)

    def forward(self, inputs, return_feature=False):
        bs = inputs.size(0)
        x = self.basemodel1.extract_features(inputs)
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        feature = x
        # x = self._dropout(x)
        x = self._fc(x)
        if return_feature:
            return x, feature
        return x

    def set_swish(self, memory_efficient=True):
        # self.basemodel1.set_swish(False)
        self.basemodel1._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for name, module in self.basemodel1.named_modules():
            if isinstance(module, MBConvBlock):
                module._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class Efficietnet_b6(nn.Module):
    def __init__(self, num_classes):
        super(Efficietnet_b6, self).__init__()
        self.basemodel1 = EfficientNet.from_pretrained('efficientnet-b6')
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(2306, num_classes)

    def forward(self, inputs):
        bs = inputs.size(0)
        x = self.basemodel1.extract_features(inputs)
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x


class Efficietnet_b7(nn.Module):
    def __init__(self, num_classes):
        super(Efficietnet_b7, self).__init__()
        self.basemodel1 = EfficientNet.from_pretrained('efficientnet-b7')
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(2560, num_classes)

    def forward(self, inputs):
        bs = inputs.size(0)
        x = self.basemodel1.extract_features(inputs)
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x


EFFICIENTNET_ZOO = {
    "b4": Efficietnet_b4,
    "b5": Efficietnet_b5,
    "b6": Efficietnet_b6,
    "b7": Efficietnet_b7,
}


def get_efficientnet(level, args):
    if level not in EFFICIENTNET_ZOO:
        raise NotImplementedError
    return EFFICIENTNET_ZOO[level](**args)


