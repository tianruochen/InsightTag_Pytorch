#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :__init__.py.py
# @Time     :2022/6/21 下午7:45
# @Author   :Chang Qing
 


import torch
from modules.loss.bce_loss import BCEWithLogitsLoss, BCELoss
from modules.loss.adaptive_selective_loss import AdaptiveSelectiveLoss

LOSS_FACTORY = {
    "bce_loss": BCELoss,
    "bce_with_logits_loss": BCEWithLogitsLoss,
    "adaptive_selective_loss": AdaptiveSelectiveLoss
}


def build_loss(loss_name, class_num, device=None):
    if loss_name not in LOSS_FACTORY:
        raise NotImplementedError
    if loss_name == "adaptive_selective_loss":
        return AdaptiveSelectiveLoss(class_num=class_num, device=device)
    return LOSS_FACTORY[loss_name](class_num=class_num)
