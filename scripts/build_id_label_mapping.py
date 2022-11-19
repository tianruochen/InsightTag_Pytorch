#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :build_id_label_mapping.py
# @Time     :2022/6/22 上午11:46
# @Author   :Chang Qing
 

from utils.comm_util import build_mapping_from_list
from utils.comm_util import save_to_json

if __name__ == '__main__':
    labels_path = "/data1/changqing/ZyMultiModal_Data/annotations/v1_cls24/v1_cls24.txt"
    labels = open(labels_path).readlines()
    labels = [label.strip() for label in labels if label]
    idx2name_map, name2idx_map = build_mapping_from_list(labels)
    print(idx2name_map)
    print(name2idx_map)

    save_path = "../configs/multimodal_v01_cls24_idx2name_cls24.json"
    save_to_json(idx2name_map, save_path)