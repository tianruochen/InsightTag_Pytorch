#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :mongo_test.py
# @Time     :2022/5/13 上午10:46
# @Author   :Chang Qing

import time
import random

from tqdm import tqdm
from utils.collection_util import MongoCollection
from utils.config_util import parse_config

if __name__ == '__main__':
    config_path = "/home/work/changqing/Insight_Multimodal_Pytorch/configs/multimodal_online_infer_pipe.yaml"
    config = parse_config(config_path)
    database = config.database
    pids = open("../../data/infer_pids/infer_pids.txt").readlines()
    pids = [pid.strip() for pid in pids if pid]
    db_src_params = database['db_src_params'].copy()
    db_dst_params = database['db_dst_params'].copy()
    src_collection = MongoCollection(db_src_params)
    dst_collection = MongoCollection(db_dst_params)

    samples = dst_collection.query_items()
    for sample in samples[-5:]:
        print(sample)
        # del_conditions = {'_id': sample['_id']}
        # dst_collection.delete_item(del_conditions)
    print(f"output库剩余items数量: {len(samples)}")

    # samples = src_collection.query_items()
    # print(len(samples))
    # for sample in tqdm(samples):
    #     del_conditions = {'_id': sample['_id']}
    #     src_collection.delete_item(del_conditions)
    # samples = src_collection.query_items()
    # print(f"input库剩余items数量: {len(samples)}")

    # print(len(pids))
    # for pid in pids[:100]:
    #     ct = int(time.time())
    #     item = {"_id": pid,
    #             "ct": ct,
    #             "processed": False
    #             }
    #     src_collection.insert_items(item)
    # for pid in pids[300:600]:
    #     sleep_time = 2 * random.random()
    #     time.sleep(sleep_time)
    #     ct = int(time.time())
    #     item = {"_id": pid,
    #             "ct": ct,
    #             "processed": False
    #             }
    #     src_collection.insert_items(item)


