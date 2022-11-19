#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :extract_videos_feature.py
# @Time     :2022/6/13 上午11:35
# @Author   :Chang Qing

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import numpy as np

from tqdm import tqdm
from glob import glob
from modules.prepare.vit_video_feature_tool import VitVideoFeatureExtractor
from utils.comm_util import ImageVideoCollector

if __name__ == '__main__':

    videos_root = "/data02/tabsun/post_tag/val_videos/"
    videos_frame_root = "/data02/changqing/ZyMultiModal_Data/videos_frame"
    # videos_feature_root = "/data02/changqing/ZyMultiModal_Data/videos_feature/vit"
    videos_feature_root = "/data02/changqing/ZyMultiModal_Data/videos_feature_fast"

    # videos_path = glob(os.path.join(videos_root, "*.mp4"))
    video_collector = ImageVideoCollector(collect_type="video")
    videos_path = video_collector.collect(videos_root)
    # print(len(videos_path))
    # print(videos_path[:5])
    # videos_path = ["/data02/tabsun/post_tag/val_videos/196470/1964707963.mp4",
    #                "/data02/tabsun/post_tag/val_videos/188485/1884857123.mp4",
    #                "/data02/tabsun/post_tag/val_videos/188289/1882895936.mp4"]
    video_feature_extractor = VitVideoFeatureExtractor(videos_frame_root=videos_frame_root)

    for video_path in tqdm(videos_path):
        # print(video_path)
        base_name = os.path.basename(video_path)
        base_id = base_name.split(".")[0]
        video_feature_path = os.path.join(videos_feature_root, base_id + ".npy")
        if os.path.exists(video_feature_path):
            continue

        video_features = video_feature_extractor.extract_features(video_path)
        print(video_path, video_features.shape)
        np.save(video_feature_path, video_features)
