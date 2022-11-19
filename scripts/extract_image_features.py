#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :extract_audio_features.py
# @Time     :2022/6/21 上午11:23
# @Author   :Chang Qing


import sys, os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

sys.path.append(os.getcwd())
import json
import time
import argparse
import random
import traceback
sys.path.append("..")
import numpy as np
from tqdm import tqdm
from glob import glob
from modules.prepare.image_feature_tool import ImageFeatureExtractor
from utils.comm_util import ImageVideoCollector


def process_dir(image_dir, image_feature_extractor, image_npy_folder):
    image_collector = ImageVideoCollector(collect_type="image")
    images_path = image_collector.collect(image_dir)
    images_nums = len(images_path)

    # images_path = sorted(glob(os.path.join(image_dir, "*.jpg")))
    # print(len(images_path))
    # print(images_path[:5])
    while True:
        for image_path in tqdm(images_path):
            if not image_path:
                continue
            base_name = os.path.basename(image_path).replace(".jpg", ".npy")
            image_npy_path = os.path.join(image_npy_folder, base_name)
            if os.path.exists(image_npy_path):
                continue
            # image_feature_extractor是按照帖子维度来处理的，所以要封装成list
            image_paths = [image_path]
            image_features = image_feature_extractor.extract_features(image_paths)
            # image_features = image_features.squeeze()
            if image_features is not None:
                # print(image_features.shape)
                np.save(image_npy_path, image_features)
        time.sleep(10)
        images_path = image_collector.collect(image_dir)
        new_images_nums = len(images_path)
        if new_images_nums == images_nums:
            return
        else:
            images_nums = new_images_nums


def process_json(json_file, image_feature_extractor, image_npy_folder):
    data = json.load(open(data_path))
    for inv_id, inv_info in tqdm(json_file.items(), total=len(data)):
        image_paths = inv_info["image_paths"]
        if not image_paths:
            continue
        image_npy_path = os.path.join(image_npy_folder, f"{inv_id}.npy")
        image_features = image_feature_extractor.extract_features(image_paths)
        np.save(image_npy_path, image_features)


if __name__ == '__main__':
    # 提取图片特征（按照帖子的维度来）
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="", type=str)
    parser.add_argument('--image_dir', default="/data02/tabsun/post_tag/support_images", type=str)
    parser.add_argument('--postfix', default='jpg', type=str)
    # parser.add_argument('--frame_npy_folder', default='%s/frame_npy'%base_path, type=str)
    parser.add_argument('--base_model_name', default='vit', type=str)
    parser.add_argument('--image_npy_folder', default='/data02/changqing/ZyMultiModal_SuppData/images_feature/vit1x1024', type=str)

    args = parser.parse_args()
    data_path = args.data_path
    image_dir = args.image_dir
    base_model_name = args.base_model_name
    image_npy_folder = args.image_npy_folder

    image_collector = ImageVideoCollector(collect_type="image")
    images_path = image_collector.collect(image_dir)
    images_nums = len(images_path)
    print(images_path[:5])
    print(images_nums)

    # assert data_path or image_dir, "data_path or image_dir is necessary!"
    #
    # os.makedirs(args.image_npy_folder, exist_ok=True)
    #
    # image_feature_extractor = ImageFeatureExtractor(base_model_name=base_model_name)
    #
    # if data_path:
    #     process_json(data_path, image_feature_extractor, image_npy_folder)
    # else:
    #     process_dir(image_dir, image_feature_extractor, image_npy_folder)
