#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :data_nums_check.py
# @Time     :2022/10/20 下午4:53
# @Author   :Chang Qing

import sys
sys.path.append("../../")
from utils.collection_util import MongoCollection
from utils.config_util import parse_config

if __name__ == '__main__':
    config_path = "/home/work/changqing/Insight_Multimodal_Pytorch/configs/multimodal_online_infer_pipe.yaml"
    config = parse_config(config_path)
    database = config.database
    db_src_params = database['db_src_params'].copy()
    db_dst_params = database['db_dst_params'].copy()
    src_collection = MongoCollection(db_src_params)
    dst_collection = MongoCollection(db_dst_params)
    samples = src_collection.query_items()
    print(f"输入库的长度为: {len(samples)}")
    samples = dst_collection.query_items()
    print(f"输出库的长度为: {len(samples)}")



