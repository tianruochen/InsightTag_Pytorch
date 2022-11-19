#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :multi_dataset.py
# @Time     :2022/6/21 下午8:12
# @Author   :Chang Qing

import os
import json
import math
import ijson
import torch
import random
import traceback
import numpy as np

from torch.utils.data import Dataset


def preprocess_frame(video, max_frames, unsqueeze_dim=False):
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
    if unsqueeze_dim:
        video_out = torch.tensor(video_out, dtype=torch.float32).unsqueeze(dim=0).cuda()
        mask = torch.tensor(np.array(mask)).unsqueeze(dim=0).cuda()
    else:
        video_out = torch.tensor(video_out, dtype=torch.float32).cuda()
        mask = torch.tensor(np.array(mask)).cuda()
    return video_out, mask


# 视频、语音、图像、文本4模态 融合dataset
class MultiModalDataset(Dataset):

    def __init__(self, data_path, device, dim=1024, num_classes=301):
        self.data_path = data_path
        self.data_info_list = self._load_data_info()
        self.data_nums = len(self.data_info_list)
        print(f"{self.data_path} nums: {len(self.data_info_list)}")
        self.device = device
        self.dim = dim
        self.num_classes = num_classes

    def _load_data_info(self):
        try:
            data_info_list = json.load(open(self.data_path))
        except:
            data_info_list = []
            with open(self.data_path, "r") as f:
                records = ijson.items(f, "item")
                for record in records:
                    data_info_list.append(record)
        return data_info_list

    def _package_video_image_features(self, item):
        image_feature_paths = item["image_feature_paths"]
        video_feature_paths = item["video_feature_paths"]

        have_image_feature = len(image_feature_paths) > 0
        have_video_feature = len(video_feature_paths) > 0
        if not have_image_feature and not have_video_feature:
            image_video_features = np.zeros((1, self.dim))
            image_video_features, image_video_mask = preprocess_frame(image_video_features, 210)
            return image_video_features, image_video_mask

        # 所有图像的特征封装为 30 * self.dim 的ndarray
        image_features = np.zeros((1, self.dim))
        if have_image_feature:
            random.shuffle(image_feature_paths)
            image_features = np.zeros((30, self.dim))
            images_nums = len(image_feature_paths)
            seg_size = math.ceil(30 / images_nums)
            for i in range(images_nums):
                image_features[i:min(i + seg_size, 30), :] = np.load(image_feature_paths[i])

        videos_features = np.zeros((0, self.dim))
        if have_video_feature:
            random.shuffle(video_feature_paths)
            for video_feature_path in video_feature_paths:
                if os.path.exists(video_feature_path):
                    videos_features = np.load(video_feature_path)
                    break

        # 如果有视频特征没有图像特征，则将视频第一帧特征作为图像特征
        if image_features.shape[0] == 1:
            image_features = np.zeros((30, self.dim))
            image_features[:] = videos_features[0]

        image_video_features = np.concatenate((image_features, videos_features), axis=0)
        image_video_features, image_video_mask = preprocess_frame(image_video_features, 210)

        return image_video_features, image_video_mask

    def _build_labels(self, item):
        if "pos_labels_idx" in item and "neg_labels_idx" in item:
            label_vector = -1 * torch.ones(self.num_classes).to(self.device)
            pos_labels = item["pos_labels_idx"]
            neg_labels = item["neg_labels_idx"]
            # print(img_path, pos_labels, neg_labels)
            label_vector[pos_labels] = 1
            label_vector[neg_labels] = 0
        else:
            labels_idx = [int(label_idx) for label_idx in item["labels_idx"]]
            label_vector = torch.zeros(self.num_classes).to(self.device)
            label_vector[labels_idx] = 1.
        return label_vector

    def __len__(self):
        return len(self.data_info_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(self.data_info_list[idx])
        item = self.data_info_list[idx]
        inv_id = item["pid"]

        try:
            # 将图像视作视频帧 进行处理，整合成一个(300, dim)的ndarray, 前30帧是图片特征，后面是视频特征。
            # 采用vit时， dim=1024; 采用tsn+eff时， dim=2048
            image_video_, image_video_mask = self._package_video_image_features(item)
            # print(image_video_.shape)
            # print(image_video_mask.shape)
            image_video_ = [image_video_, image_video_mask]

            audio_ = np.zeros((1, 1024))
            audio_feature_paths = item["audio_feature_paths"]
            if audio_feature_paths and os.path.exists(audio_feature_paths[0]):
                audio_ = np.load(audio_feature_paths[0])
            audio_, audio_mask = preprocess_frame(audio_, 120)
            audio_ = audio_[:, :128]
            # print(audio_.shape)
            # print(audio_mask.shape)
            audio_ = [audio_, audio_mask]

            # token_ids, seq_len, mask
            # 帖子内容
            text_feature = json.load(open(item["text_feature_path"]))
            text_content_feature = text_feature["content_feature"]
            # ocr识别结果
            ocr_feature = json.load(open(item["ocr_feature_path"]))
            ocr_content_feature = ocr_feature["content_feature"]
            # topic_feature = text_feature["topic_feature"]

            text_ = [
                (torch.LongTensor(text_content_feature["token_ids"]).to(self.device),
                 torch.tensor(int(text_content_feature["seq_len"])).to(self.device),
                 torch.tensor(np.array(text_content_feature["mask"])).to(self.device)),
                (torch.LongTensor(ocr_content_feature["token_ids"]).to(self.device),
                 torch.tensor(int(ocr_content_feature["seq_len"])).to(self.device),
                 torch.tensor(np.array(ocr_content_feature["mask"])).to(self.device))
            ]

            # semi_labels = True 使用部分标签
            label_ = self._build_labels(item)
            # print(label_.shape)
            # print(text_)
            return text_, image_video_, audio_, label_, inv_id
        except:
            traceback.print_exc()
            print(inv_id)
            return self.__getitem__(idx + 1)


if __name__ == '__main__':
    image_video_features = np.zeros((1, 1024))
    image_video_features, image_video_mask = preprocess_frame(image_video_features, 210)
    print(image_video_features)
    print(image_video_mask)
