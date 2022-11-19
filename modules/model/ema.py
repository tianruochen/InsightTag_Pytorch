# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :ema.py
# @Time     :2021/9/1 下午3:51
# @Author   :Chang Qing

import torch

from copy import deepcopy


class ModelEMA(object):
    def __init__(self, model, decay, device):
        self.ema = deepcopy(model)
        self.ema.to(device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, "module")

        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]

        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        need_module = hasattr(model, "module") and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()

            # update parameters
            for k in self.param_keys:
                if need_module:
                    j = "module." + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            # update buffers
            for k in self.buffer_keys:
                if need_module:
                    j = "module." + k
                else:
                    j = k
                esd[k].copy_(msd[j])




