#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :base.py
# @Time     :2022/6/21 下午8:38
# @Author   :Chang Qing


import json
import logging
import os
import random
from collections import OrderedDict

import numpy as np
import torch

from modules.model.fusion_Bi_trans_vit import MultiFusionNet
# from modules.model.fusion_Bi_trans_vit1 import MultiFusionNet


class Base:

    def __init__(self, basic, arch):
        self.basic = basic
        self.arch = arch

        # basic
        self.name = self.basic.name
        self.version = self.basic.version
        self.task_name = self.basic.task_name
        self.task_type = self.basic.task_type
        self.seed = self.basic.seed
        self.n_gpus = self.basic.n_gpus
        self.id2name_path = self.basic.id2name

        # set random seed
        if self.seed:
            self._fix_random_seed()

        # set up logger
        self.logger = self._setup_logger()

        # set up device
        self.device, self.device_ids = self._setup_device(self.n_gpus)
        self.num_classes = self.arch.args.num_classes

        # build id to name mapping
        self.id2name = self._build_label_class(self.id2name_path)
        assert self.num_classes == len(
            self.id2name), f"ERROR! cls num not match {self.num_classes} != {len(self.id2name)}"


        # build model
        self.model = MultiFusionNet(**self.arch.args)

        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        # load checkpoint
        if self.arch.get("resume", None):
            self.logger.info("resume checkpoint...")
            self._resume_checkpoint(self.arch.resume)
        elif self.arch.get("best_model", None):
            self.logger.info("load best model.....")
            self._load_best_model(self.arch.best_model)


    def _fix_random_seed(self):
        random.seed(self.seed)
        # torch.random.seed()
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # torch.cuda.manual_seed(self.seed)

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            # format="[%(asctime)12s] [%(levelname)7s] (%(filename)15s:%(lineno)3s): %(message)s",
            format="[%(asctime)12s] [%(levelname)s] : %(message)s",
            handlers=[
                logging.StreamHandler(),
            ]
        )
        return logging.getLogger(self.__class__.__name__)

    def _setup_device(self, n_gpu_need):
        """
            check training gpu environment
            :param n_gpu_need: int
            :return:
            """
        n_gpu_available = torch.cuda.device_count()
        gpu_list_ids = []

        if n_gpu_available == 0 and n_gpu_need > 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        elif n_gpu_need > n_gpu_available:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                    n_gpu_need, n_gpu_available))
            n_gpu_need = n_gpu_available
            self.n_gpus = n_gpu_available
        if n_gpu_need == 0:
            self.logger.info("run models on CPU.")
        else:
            logging.info(f"run model on {n_gpu_need} gpu(s)")
            # gpu_list_str = os.environ["CUDA_VISIBLE_DEVICES"]
            # gpu_list_ids = [int(i) for i in gpu_list_str.split(",")][:n_gpu_need]
            gpu_list_ids = [int(i) for i in range(n_gpu_need)]
        device = torch.device("cuda" if n_gpu_need > 0 else "cpu")
        return device, gpu_list_ids

    def _build_label_class(self, label2name_path):
        id2name = {}
        if os.path.exists(label2name_path):
            id2name = json.loads(open(label2name_path).read())
            return id2name
            # if len(label2name.keys()) == self.cls_num:
            #     return label2name
            # else:
            #     raise ValueError("classes num error")
        else:
            assert self.num_classes, "class num is not explicit!"
            for i in range(self.num_classes):
                id2name[str(i)] = "class_" + str(i)
            return id2name

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {}".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.arch:
            self.logger.warning(
                'Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
                'This may yield an exception while state_dict is being loaded.')
        if self.n_gpus > 1:
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if not k.startswith("module."):
                    break
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict if new_state_dict else checkpoint['state_dict']
            self.model.load_state_dict(state_dict, strict=True)

        # resume ema model
        if self.use_ema:
            self.ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])

        # # load optimizer state from checkpoint only when optimizer type is not changed.
        # if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
        # 	self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
        # 						'Optimizer parameters not being resumed.')
        # else:
        # 	self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.results_log = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch - 1))

    def _load_best_model(self, best_model_path):
        """
        load best model (only weight)
        :param best_model_path:
        :return:
        """
        print(best_model_path)
        checkpoint = torch.load(best_model_path)
        # remove _fc or not
        strict = True
        if list(checkpoint.values())[-1].shape[0] != self.num_classes:
            checkpoint.pop(list(checkpoint.keys())[-1])  # pop module._fc.bias
            checkpoint.pop(list(checkpoint.keys())[-1])  # pop module._fc.weight
            strict = False
            self.logger.info("** Not load FC layer **")
        if self.n_gpus > 1:
            self.model.load_state_dict(checkpoint, strict=strict)
        else:
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                if not k.startswith("module."):
                    break
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict if new_state_dict else checkpoint
            self.model.load_state_dict(state_dict, strict=strict)
        # self.model.load_state_dict(torch.load(best_model_path))

