#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :build_trainval_data.py
# @Time     :2022/6/14 下午4:34
# @Author   :Chang Qing


# 已废弃不用

import os
import json
import random

videos_feature_root = "/data1/changqing/ZyMultiModal_Data/videos_feature"
audios_feature_root = "/data1/changqing/ZyMultiModal_Data/audios_feature"
text_feature_root = "/data1/changqing/ZyMultiModal_Data/texts_feature"

focus_labels = ["游戏**CF", "游戏**CSGO", "游戏**绝地求生", "游戏**桌游卡牌", "游戏**其他",
                "影视**古装", "影视**战争", "影视**灾难", "影视**科幻", "影视**动画片",
                "知识科普**知识科普", "知识科普**历史解读",
                "截图录屏**最右截图录屏",
                "校园**宿舍", "校园**教室", "校园**网课",
                "秀身材**秀胸", "秀身材**秀腰臀腿",
                "美食**小零食",
                "软件软体**其他",
                "生活**直播",
                "资源站**小说网文",
                "服饰配饰**男装", "服饰配饰**女装"]


def supplement_data(data, mode):
    new_data = {}
    for inv_id, inv_info in data.items():

        # 补充视频、音频特征地址信息
        videos_info = inv_info["videos_info"]
        video_feature_path = ""
        audio_feature_path = ""
        for video_id, _ in videos_info.items():
            f_name = f"{inv_id}_{video_id}.npy"
            v_f_path = os.path.join(videos_feature_root, f_name)
            a_f_path = os.path.join(audios_feature_root, f_name)
            if os.path.isfile(v_f_path) and os.path.isfile(a_f_path):
                video_feature_path = v_f_path
                audio_feature_path = a_f_path
                break

        # 补充文本特征地址信息
        text_feature_path = ""
        no_text_feature_path = "/data1/changqing/ZyMultiModal_Data/texts_feature/no_data.json"
        t_f_path = os.path.join(text_feature_root, f"{inv_id}.json")
        if os.path.isfile(t_f_path):
            text_feature_path = t_f_path
        else:
            text_feature_path = no_text_feature_path

        inv_info["text_feature_path"] = text_feature_path
        inv_info["video_feature_path"] = video_feature_path
        inv_info["audio_feature_path"] = audio_feature_path

        if text_feature_path == no_text_feature_path and not video_feature_path and \
                not audio_feature_path and mode == "class":
            continue
        new_data[inv_id] = inv_info

    return new_data


def split_trainval_class(data):
    train_data = []
    valid_data = []
    train_inv_ids = []
    valid_inv_ids = []
    for focus_label in focus_labels:
        one_class_ids = []
        for inv_id, inv_info in data.items():
            labels = inv_info["labels"]
            if focus_label in labels:
                one_class_ids.append(inv_id)
        nums = len(one_class_ids)
        print(f"{focus_label}--{nums}")
        random.shuffle(one_class_ids)
        train_inv_ids.extend(one_class_ids[:int(nums * 0.85)])
        valid_inv_ids.extend(one_class_ids[int(nums * 0.85):])

    for train_id in train_inv_ids:
        inv_info = data[train_id]
        labels_ids = []
        labels = inv_info["labels"]
        for label in labels:
            labels_ids.append(focus_labels.index(label))
        train_info = {
            "id": train_id,
            "text_feature_path": inv_info["text_feature_path"],
            "video_feature_path": inv_info["video_feature_path"],
            "audio_feature_path": inv_info["audio_feature_path"],
            "labels": labels_ids
        }
        train_data.append(train_info)

    for valid_id in valid_inv_ids:
        inv_info = data[valid_id]
        labels_ids = []
        labels = inv_info["labels"]
        for label in labels:
            labels_ids.append(focus_labels.index(label))
        valid_info = {
            "id": valid_id,
            "text_feature_path": inv_info["text_feature_path"],
            "video_feature_path": inv_info["video_feature_path"],
            "audio_feature_path": inv_info["audio_feature_path"],
            "labels": labels_ids
        }
        valid_data.append(valid_info)

    return train_data, valid_data


def split_trainval_other(data, limit_nums=0):
    train_data = []
    valid_data = []
    train_inv_ids = []
    valid_inv_ids = []

    total_other_ids = list(data.keys())
    random.shuffle(total_other_ids)
    if limit_nums:
        total_other_ids = total_other_ids[:limit_nums]
    total_other_nums = len(total_other_ids)
    train_inv_ids.extend(total_other_ids[:int(total_other_nums * 0.85)])
    valid_inv_ids.extend(total_other_ids[int(total_other_nums * 0.85):])

    for train_id in train_inv_ids:
        inv_info = data[train_id]
        labels_ids = []

        train_info = {
            "id": train_id,
            "text_feature_path": inv_info["text_feature_path"],
            "video_feature_path": inv_info["video_feature_path"],
            "audio_feature_path": inv_info["audio_feature_path"],
            "labels": labels_ids
        }
        train_data.append(train_info)

    for valid_id in valid_inv_ids:
        inv_info = data[valid_id]
        labels_ids = []
        valid_info = {
            "id": valid_id,
            "text_feature_path": inv_info["text_feature_path"],
            "video_feature_path": inv_info["video_feature_path"],
            "audio_feature_path": inv_info["audio_feature_path"],
            "labels": labels_ids
        }
        valid_data.append(valid_info)

    return train_data, valid_data


if __name__ == '__main__':
    # 具体类别的处理
    input_path = "/data1/changqing/ZyMultiModal_Data/annotations/v1_cls24/filtered_invs_data_20210101_20220430_cls24.json"
    input_data = json.load(open(input_path))
    input_data = supplement_data(input_data, mode="class")
    train_class_data, valid_class_data = split_trainval_class(input_data)
    print(f"train nums for class: {len(train_class_data)}")
    print(f"valid nums for class: {len(valid_class_data)}")

    # 其他类别的处理
    other_path = "/data1/changqing/ZyMultiModal_Data/annotations/v1_cls24/filtered_invs_data_20220101_20220430_cls24_others.json"
    other_data = json.load(open(other_path))
    other_data = supplement_data(other_data, mode="other")
    train_other_data, valid_other_data = split_trainval_other(other_data)
    print(f"train nums for other: {len(train_other_data)}")
    print(f"valid nums for other: {len(train_other_data)}")

    train_data = train_class_data + train_other_data
    random.shuffle(train_data)
    train_data_path = "/data1/changqing/ZyMultiModal_Data/annotations/v1_cls24/train_cls24.json"
    json.dump(train_data, open(train_data_path, "w"), ensure_ascii=False, indent=4)

    valid_data = valid_class_data + valid_other_data
    random.shuffle(valid_data)
    valid_data_path = "/data1/changqing/ZyMultiModal_Data/annotations/v1_cls24/valid_cls24.json"
    json.dump(valid_data, open(valid_data_path, "w"), ensure_ascii=False, indent=4)

    print(f"train nums: {len(train_data)}")
    print(f"valid nums: {len(valid_data)}")
    print("done")
