#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :catch_fpfn.py
# @Time     :2022/9/15 上午11:12
# @Author   :Chang Qing

import json
import numpy as np
import pandas as pd
from tqdm import tqdm


def catch_cate_pids(data, cate_name):
    pids = []
    for item in tqdm(data):
        labels = item["labels_name"]
        if cate_name in labels:
            pid = item["pid"]
            pids.append(pid)
    return pids


def catch_fpfn_by_results(res_file, category_names=None):
    # res_file = "../../experiments/v2_cls301/predict_output_all.csv"

    # videos_pid = json.load(open(videos_pid_path))
    if category_names is None:
        category_names = []
    data = pd.read_csv(res_file)

    # category_name = "财经_股票基金"
    for category_name in category_names:
        fp_list = []
        fn_list = []
        fp_pids = []
        fn_pids = []
        total = 0
        match = 0
        for index, row in data.iterrows():
            pid = str(row["pids"])
            # print(row["gt_labels"])
            # print(row["pd_labels"])
            gt_label = str(row["gt_labels"]).split(",")
            pd_label = str(row["pd_labels"]).split(",")
            # print(gt_label)
            # print(pd_label)
            # if pid not in videos_pid:
            #     continue
            if category_name in gt_label:
                total += 1
                if category_name in pd_label:
                    match += 1
            if category_name in gt_label and category_name not in pd_label:
                fn_pids.append(pid)
                fn_list.append(f"{pid}  gt_label: {gt_label}, pd_label: {pd_label}")
            if category_name in pd_label and category_name not in gt_label:
                fp_pids.append(pid)
                fp_list.append(f"{pid}  gt_label: {gt_label}, pd_label: {pd_label}")

        print(f"Total nums: {total}, Match: {match}")
        print("=" * 100)
        print(f"FN nums: {len(fn_list)}")
        print(" ".join(fn_pids))
        print("\n".join(fn_list))
        print("=" * 100)
        print(f"FP nums: {len(fp_list)}")
        print(" ".join(fp_pids))
        print("\n".join(fp_list))


def get_pids():
    pids = []
    file_path = "/data02/changqing/ZyMultiModal_Data/annotations/v2_cls301/test_cls301_13w.json"
    data = json.load(open(file_path))
    for item in data:
        pid = item["pid"]
        pids.append(pid)
    return pids


def catch_fpfn_by_scores(scores_matrix_path, labels_matrix_path, name2idx_path):
    scores_matrix = np.load(scores_matrix_path)
    labels_matrix = np.load(labels_matrix_path)
    pids = get_pids()
    name2idx = json.load(open(name2idx_path, "r"))
    category_names = list(name2idx.keys())
    for category_name in category_names:
        idx = int(name2idx[category_name])
        scores = list(scores_matrix[:, idx])
        labels = list(labels_matrix[:, idx])
        fp_pids = []
        fn_pids = []
        for pid, score, label in zip(pids, scores, labels):
            if label == 1 and score < 0.8:
                fn_pids.append(pid)
            if label == 0 and score > 0.1:
                fp_pids.append(pid)
        diff_num = len(fn_pids) - len(fp_pids)
        if diff_num > 0:
            fp_pids += [""] * diff_num
        else:
            fn_pids += [""] * abs(diff_num)

        df = pd.DataFrame({
            "fp_pids": fp_pids,
            "fn_pids": fn_pids
        })
        df.to_csv(f"/home/work/changqing/Insight_Multimodal_Pytorch/data/check_csv_2/{category_name}.csv")
        # break


def labels_equeal(gt_labels, pd_labels):
    if len(gt_labels) != len(pd_labels):
        return False
    for gt_label, pd_label in zip(gt_labels, pd_labels):
        if gt_label != pd_label:
            return False
    return True


def catch_gtpd_labels(scores_matrix_path, labels_matrix_path, idx2name_path):
    scores_matrix = np.load(scores_matrix_path)
    labels_matrix = np.load(labels_matrix_path)
    pids = get_pids()
    idx2name = json.load(open(idx2name_path, "r"))

    pd_labels_list = []
    gt_labels_list = []
    union_labels_list = []
    file_path = "/data02/changqing/ZyMultiModal_Data/annotations/v2_cls301/test_cls301_13w.json"
    data = json.load(open(file_path))
    valid_pids = []
    for i, item in enumerate(data):
        pid = item["pid"]
        gt_labels = item["labels_name"]
        gt_labels = sorted(gt_labels)
        scores = list(scores_matrix[i])
        pd_labels = []
        for i, score in enumerate(scores):
            if score > 0.2:
                pd_label = idx2name[str(i)]
                pd_labels.append(pd_label)
        pd_labels = sorted(pd_labels)

        if labels_equeal(gt_labels, pd_labels):
            continue
        valid_pids.append(pid)
        pd_labels_list.append(",".join(pd_labels))
        gt_labels_list.append(",".join(gt_labels))
        union_labels = list(set(pd_labels).union(set(gt_labels)))
        union_labels_list.append(",".join(union_labels))

    df = pd.DataFrame({
        "pids": valid_pids,
        "gt_labels": gt_labels_list,
        "our_pd_labels": pd_labels_list,
        "gt_our_union_labels": union_labels_list
    })

    df.to_csv(f"/home/work/changqing/Insight_Multimodal_Pytorch/data/infer_pids/test_13w_pd_gt.csv")


if __name__ == '__main__':
    scores_matrix_path = "/home/work/changqing/Insight_Multimodal_Pytorch/workshop/multimodal_v01_cls301_0825_104526/res/test/scores_matrix.npy"
    labels_matrix_path = "/home/work/changqing/Insight_Multimodal_Pytorch/workshop/multimodal_v01_cls301_0825_104526/res/test/labels_matrix.npy"
    name2idx_path = "/data02/changqing/ZyMultiModal_Data/annotations/v1_cls301/name2idx_cls301.json"
    idx2name_path = "/data02/changqing/ZyMultiModal_Data/annotations/v1_cls301/idx2name_cls301.json"
    # catch_fpfn_by_scores(scores_matrix_path, labels_matrix_path, name2idx_path)
    catch_gtpd_labels(scores_matrix_path, labels_matrix_path, idx2name_path)
