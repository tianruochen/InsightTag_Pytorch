#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :data_query.py
# @Time     :2022/10/31 上午11:26
# @Author   :Chang Qing

import sys
sys.path.append("../../")

import argparse
import numpy as np

from utils.comm_util import save_to_json
from utils.collection_util import MongoCollection

collection_config = {
    "input": {
        "host": "",
        "port": 27047,
        "db": "multimodal_tag",
        "table": "multimodal_tag_input"
    },
    "output": {
        "host": "",
        "port": 27047,
        "db": "monet",
        "table": "post_labels_v3"
    }
}



def query_items(collection, conditions):
    res_items = collection.query_items(conditions=conditions)
    return res_items


def query_nums(collection, conditions):
    res = query_items(collection, conditions)
    return len(res)


def build_conditions(args):
    conditions = []
    pids = args.pids
    process_tik = args.process_tik
    process_tok = args.process_tok
    model_forward_tik = args.model_forward_tik
    model_forward_tok = args.model_forward_tok

    if pids:
        conditions += [[f"_id:=:{pid}"] for pid in pids.split(" ")]
    time_conditions = []
    time_conditions += [f"process_tik:>:{process_tik}"] if process_tik else []
    time_conditions += [f"process_tok:<:{process_tok}"] if process_tok else []
    time_conditions += [f"model_forward_tik:>:{model_forward_tik}"] if model_forward_tik else []
    time_conditions += [f"process_tok:<:{model_forward_tok}"] if model_forward_tok else []
    if time_conditions:
        conditions.append(time_conditions)
    return conditions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Query Scripts")
    parser.add_argument("--tb_name", type=str, default="output")
    parser.add_argument("--query_type", type=str, default="query_items")
    parser.add_argument("--pids", type=str, default="309831999")
    parser.add_argument("--process_tik", type=float, default=None)
    parser.add_argument("--process_tok", type=float, default=None)
    parser.add_argument("--model_forward_tik", type=float, default=None)
    parser.add_argument("--model_forward_tok", type=float, default=None)
    args = parser.parse_args()

    # args.process_tik = 1667059200               # 2022-10-30
    # args.process_tok = 1667145600               # 2022-10-31    105657
    # args.model_forward_tik = 1667059200         # 2022-10-30
    # args.model_forward_tok = 1667145600         # 2022-10-31      125497

    tb_name = args.tb_name
    tb_params = collection_config[tb_name]
    collection = MongoCollection(tb_params)
    conditions = build_conditions(args)
    # res = collection.get_batch_items(conditions, "process_tik", 100)
    # for item in res:
    #     print(item)
    # print(res)

    query_type = args.query_type
    if query_type == "query_nums":
        nums = query_nums(collection, conditions)
        print(nums)
    else:
        items = query_items(collection, conditions)
        process_cost_list = []
        model_forward_cost_list = []
        samples = {}
        for item in items:
            pid = item["_id"]
            print(item)
            # process_cost = float(item["process_cost"][:-1])
            # if process_cost > 500:
            #     print(pid, process_cost)
            # model_forward_cost = float(item["model_forward_cost"][:-1])
            # process_cost_list.append(process_cost)
            # model_forward_cost_list.append(model_forward_cost)
            samples[pid] = item

        # print(np.mean(process_cost_list))
        # print(np.mean(model_forward_cost_list))
        print(len(items))
        # save_to_json(samples, "multimodal_our_infer_2022-10-30.json")
