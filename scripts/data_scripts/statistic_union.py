#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :catch_cate_pids.py
# @Time     :2022/8/9 下午3:54
# @Author   :Chang Qing


import json
from tqdm import tqdm

"""
查看数据集之间是否有交集
"""


def get_union_nums(dataset1, dataset2):
    union_nums = 0
    dataset1_pids = {}
    for item in tqdm(dataset1):
        pid = item["pid"]
        dataset1_pids[pid] = ""

    for item in tqdm(dataset2):
        pid = item["pid"]
        if pid in dataset1_pids:
            print(pid)
            union_nums += 1
    return union_nums




if __name__ == '__main__':
    train_data_path = "/data02/changqing/ZyMultiModal_Data/annotations/v2_cls301/train_cls301_86w.json"
    valid_data_path = "/data02/changqing/ZyMultiModal_Data/annotations/v2_cls301/valid_cls301_15w.json"
    test_data_path = "/data02/changqing/ZyMultiModal_Data/annotations/v2_cls301/test_cls301_13w.json"

    train_data = json.load(open(train_data_path))
    valid_data = json.load(open(valid_data_path))
    test_data = json.load(open(test_data_path))

    print(len(train_data))
    print(len(valid_data))
    print(len(test_data))

    union_nums = get_union_nums(train_data, valid_data)
    print(union_nums)
    union_nums = get_union_nums(test_data, valid_data)
    print(union_nums)
    union_nums = get_union_nums(train_data, test_data)
    print(union_nums)


    # cate_name = "兴趣爱好_书法"
    # valid_pids = catch_cate_pids(valid_data, cate_name)
    # test_pids = catch_cate_pids(test_data, cate_name)
    # print(" ".join(valid_pids))
    # print("*" * 20)
    # print(" ".join(test_pids))
