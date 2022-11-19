#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :server_test.py
# @Time     :2022/10/12 下午7:32
# @Author   :Chang Qing


import time
import json
import traceback
import requests

import gradio as gr
from tqdm import tqdm
from utils.comm_util import save_to_json

def test(pid):
    payload = {
        "pid": pid
    }
    # resp = requests.post('http://172.18.108.122:6621/api/multimodal_tag', data=json.dumps(payload))
    resp = requests.post('http://multimodal-tag-cls301.srv.ixiaochuan.cn/api/multimodal_tag', data=json.dumps(payload))
    content = json.loads(resp.content)
    return content

def test_list(pids):
    pids = open("data/infer_pids/infer_pids.txt").readlines()
    pids = [pid.strip() for pid in pids if pid]
    results = {}
    pids = ["307363306"]
    for pid in tqdm(pids):
        try:
            res = test(pid)
            print(res)
            name_score_dict = res.get("data", {}).get("result", {})
            labels = list(name_score_dict.keys())
            # print(pid, labels)
        except:
            print(pid)
            traceback.print_exc()
            name_score_dict = {}
        results[pid] = name_score_dict
    save_to_json(results, "data/infer_pids_res.json")

if __name__ == "__main__":
    tik = time.time()

    demo = gr.Interface(fn=test, inputs="text", outputs="json")
    demo.launch(share=True)

    #     results[pid] = name_score_dict
    # save_to_json(results, "data/infer_pids_res.json")

    # data = json.load(open("data/infer_pids/infer_pids_res.json"))
    # pids = []
    # labels = []
    # for pid, info in data.items():
    #     label = list(info.keys())
    #     label = ",".join(label)
    #     pids.append(pid)
    #     labels.append(label)
    # import pandas as pd
    # df = pd.DataFrame({
    #     "pids": pids,
    #     "labels": labels
    # })
    # df.to_csv("data/infer_pids/infer_pids_res.csv")
    #
