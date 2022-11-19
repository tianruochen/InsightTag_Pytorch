#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :multi_dataset.py
# @Time     :2022/6/21 下午8:12
# @Author   :Chang Qing

import os
import json
import torch
import numpy as np

from torch.utils.data import Dataset


def preprocess_frame(video, max_frames):
    num_frames = video.shape[0]
    dim = video.shape[1]
    padding_length = max_frames - num_frames
    if padding_length > 0:
        mask = [1] * num_frames + ([0] * padding_length)
        fillarray = np.zeros((padding_length, dim))
        video_out = np.concatenate((video, fillarray), axis=0)
    else:
        video_out = video[:max_frames, ...]
        mask = [1] * max_frames
    video_out = torch.tensor(video_out, dtype=torch.float32).cuda()
    mask = torch.tensor(np.array(mask)).cuda()
    return video_out, mask


class MultiDataset(Dataset):

    def __init__(self, data_path, device):
        self.data_path = data_path
        self.data_info_list = json.load(open(self.data_path))
        self.device = device

    def __len__(self):
        return len(self.data_info_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(self.data_info_list[idx])
        inv_id = self.data_info_list[idx]["id"]
        video_ = np.zeros((1, 1024))
        # print(self.data_info_list[idx])
        if os.path.exists(self.data_info_list[idx]["video_feature_path"]):
            video_ = np.load(self.data_info_list[idx]["video_feature_path"])

        video_, video_mask = preprocess_frame(video_, 300)
        # else:
        #     mask = [1] + [0] * 299
        #     # video_ = torch.tensor(video_, dtype=torch.float32).cuda()
        #
        #     video_, video_mask = preprocess_frame(video_, 300)
        #     video_mask = torch.tensor(np.array(mask)).cuda()

        # video_mask = torch.tensor(np.array(mask)).cuda()

        video_ = [video_, video_mask]

        audio_ = np.zeros((1, 1024))
        if os.path.exists(self.data_info_list[idx]["audio_feature_path"]):
            audio_ = np.load(self.data_info_list[idx]["audio_feature_path"])
        audio_, audio_mask = preprocess_frame(audio_, 120)
        audio_ = audio_[:, :128]
        # print(audio_.shape)
        # print(audio_mask.shape)
        audio_ = [audio_, audio_mask]

        # token_ids, seq_len, mask
        text_feature = json.load(open(self.data_info_list[idx]["text_feature_path"]))
        content_feature = text_feature["content_feature"]
        topic_feature = text_feature["topic_feature"]

        text_ = [(torch.LongTensor(content_feature["token_ids"]).to(self.device),
                  torch.tensor(int(content_feature["seq_len"])).to(self.device),
                  torch.tensor(np.array(content_feature["mask"])).to(self.device)),
                 (torch.LongTensor(topic_feature["token_ids"]).to(self.device),
                  torch.tensor(int(topic_feature["seq_len"])).to(self.device),
                  torch.tensor(np.array(topic_feature["mask"])).to(self.device))]

        label_index = self.data_info_list[idx]["labels"]
        label_ = torch.zeros(24).to(self.device)
        label_[label_index] = 1.

        # print(label_.shape)
        # print(text_)
        return text_, video_, audio_, label_, inv_id
