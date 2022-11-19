#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :zyvideo_reader.py.py
# @Time     :2021/1/20 上午11:23
# @Author   :Chang Qing


import random
try:
    import cPickle as pickle
    from cStringIO import StringIO
except ImportError:
    import pickle
    from io import BytesIO
import numpy as np

from .reader_utils import DataReader
from .kinetics_reader import mp4_loader, imgs_transform



class ZYVideoReader(DataReader):

    def __init__(self, name, mode, cfg):
        super(ZYVideoReader, self).__init__(name, mode, cfg)
        self.format = cfg.MODEL.format
        self.num_classes = self.get_config_from_sec('model', 'num_classes')
        self.seg_num = self.get_config_from_sec('model', 'seg_num')
        self.seglen = self.get_config_from_sec('model', 'seglen')

        self.seg_num = self.get_config_from_sec(mode, 'seg_num', self.seg_num)
        self.short_size = self.get_config_from_sec(mode, 'short_size')
        self.target_size = self.get_config_from_sec(mode, 'target_size')
        self.num_reader_threads = self.get_config_from_sec(mode,
                                                           'num_reader_threads')
        self.buf_size = self.get_config_from_sec(mode, 'buf_size')
        self.fix_random_seed = self.get_config_from_sec(mode, 'fix_random_seed')

        self.img_mean = np.array(cfg.MODEL.image_mean).reshape(
            [3, 1, 1]).astype(np.float32)
        self.img_std = np.array(cfg.MODEL.image_std).reshape(
            [3, 1, 1]).astype(np.float32)
        # set batch size and file list
        self.batch_size = cfg[mode.upper()]['batch_size']
        self.filelist = cfg[mode.upper()]['filelist']

        if self.fix_random_seed:
            random.seed(0)
            np.random.seed(0)
            self.num_reader_threads = 1

    def decode(self, filepath):
        images = mp4_loader(filepath, self.seg_num, self.seglen, self.mode)
        images_transform = imgs_transform(images, self.mode, self.seg_num, self.seglen, self.short_size,
                                          self.target_size, self.img_mean, self.img_std, self.name)
        # print(images_transform.shape)
        return images_transform

