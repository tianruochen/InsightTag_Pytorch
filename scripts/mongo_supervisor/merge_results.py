#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :merge_results.py
# @Time     :2022/10/31 下午7:51
# @Author   :Chang Qing

import json
import pandas as pd



if __name__ == '__main__':
    our_predict = json.load(open("multimodal_our_infer_2022-10-30.json"))
    zl0_predict = json.load(open("multimodal_online_infer_2022-10-30_auto0.json"))
    zl1_predict = json.load(open("multimodal_online_infer_2022-10-30_auto1.json"))

    flag_list = []
    pid_list = []
    zh_pred_list = []
    our_pred_list = []
    for pid, item in zl0_predict.items():
        pid_list.append(pid)
        zh_pred = item["labels"]
        zh_pred = ",".join(sorted(zh_pred))
        # print(pid)
        our_pred = our_predict.get(pid, {}).get("results", {})
        if our_pred:
            our_pred = list(our_pred.keys())
        else:
            our_pred = []
        our_pred = ",".join(sorted(our_pred))
        zh_pred_list.append(zh_pred)
        our_pred_list.append(our_pred)
        flag_list.append("人工打标")

    for pid, item in zl1_predict.items():
        pid_list.append(pid)
        zh_pred = item["labels"]
        zh_pred = ",".join(sorted(zh_pred))
        # our_pred = list(our_predict.get(pid, {}).get("results", {}).keys())
        our_pred = our_predict.get(pid, {}).get("results", {})
        if our_pred:
            our_pred = list(our_pred.keys())
        else:
            our_pred = []
        our_pred = ",".join(sorted(our_pred))
        zh_pred_list.append(zh_pred)
        our_pred_list.append(our_pred)
        flag_list.append("张磊模型")

    df = pd.DataFrame({
        "pids": pid_list,
        "flags": flag_list,
        "zl_preds": zh_pred_list,
        "our_preds": our_pred_list
    })

    df.to_csv("/home/work/changqing/Insight_Multimodal_Pytorch/data/online_vs_ours/online_vs_ours_20221030.csv")

