#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :train_net.py.py
# @Time     :2022/6/21 下午5:41
# @Author   :Chang Qing

"""
端到端的处理
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import json
import math
import glob
import torch
import random
import pprint
import argparse

import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tqdm import tqdm
from utils.config_util import parse_config
from utils.config_util import merge_config
from modules.solver.inferer import Inferer
from utils.comm_util import save_to_json, save_to_txt
from utils.zydata_util import get_inv_info_by_pid, download_file
from modules.prepare import TextFeatureExtractor, ImageFeatureExtractor
from modules.prepare import VitVideoFeatureExtractor, AudioFeatureExtractor


def build_infer_data_test():
    data_path = "/data02/changqing/ZyMultiModal_Data/annotations/v2_cls301/test_cls301_13w.json"
    data = json.load(open(data_path))[5]

    print(data)

    pid = data["pid"]
    text_feature = json.load(open(data["text_feature_path"]))["content_feature"]
    image_feature_paths = data["image_feature_paths"]
    video_feature = np.load(data["video_feature_paths"][0])
    audio_feature = np.load(data["audio_feature_paths"][0])

    images_nums = len(image_feature_paths)
    image_feature = np.zeros((images_nums, 1024))

    for i in range(images_nums):
        image_feature[i, :] = np.load(image_feature_paths[i])

    infer_data = {
        "pid": pid,
        "text_feature": text_feature,
        "image_feature": image_feature,
        "audio_feature": audio_feature,
        "video_feature": video_feature
    }
    return infer_data


def parse_images_info(images_info, temp_dir):
    images_dir = os.path.join(temp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    images_path = []
    for image_id, image_info in images_info.items():
        image_url = image_info["img_url"]
        image_path = download_file(str(image_id), image_url, images_dir, type="image")
        if image_path:
            images_path.append(image_path)
    return images_path


def parse_videos_info(videos_info, temp_dir, one_enough=True):
    videos_dir = os.path.join(temp_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    videos_path = []
    for video_id, video_info in videos_info.items():
        video_url = video_info["video_url"]
        video_path = download_file(str(video_id), video_url, videos_dir, type="video")
        if video_path and one_enough:
            return video_path
        elif video_path:
            videos_path.append(video_path)
    return videos_path


def build_infer_data(inv_info, text_feature_extractor, image_feature_extractor, audio_feature_extractor,
                     video_feature_extractor, temp_dir):
    """
    解析input_data, 下载数据，提取特征，构建前向推理文件
    :param text_feature_extractor:
    :param input_data:
    :return: infer_data
    """
    pid = inv_info["pid"]
    text_content = inv_info["text_content"]
    images_info = inv_info["images_info"]
    videos_info = inv_info["videos_info"]

    inv_dir = os.path.join(temp_dir, pid)

    # 提取文本特征
    text_feature = text_feature_extractor.extract_features(text_content)

    # 解析图片信息，并提取图片特征
    image_paths = parse_images_info(images_info, temp_dir=inv_dir)

    # 解析视频信息，并提取视频和音频信息

    video_path = parse_videos_info(videos_info, temp_dir=inv_dir, one_enough=True)
    image_feature = None
    if image_paths:
        image_feature = image_feature_extractor.extract_features(image_paths=image_paths, remove_image=False)
    audio_feature = None
    video_feature = None
    if video_path:
        audio_feature = audio_feature_extractor.extract_features(video_path=video_path, temp_audios_dir=inv_dir,
                                                                 remove_audio=False)
        video_frame_dir = os.path.join(inv_dir, "video_frames")
        video_feature = video_feature_extractor.extract_features(video_path=video_path, video_frame_dir=video_frame_dir,
                                                                 remove_video=False)
    infer_data = {
        "pid": pid,
        "text_feature": text_feature,
        "image_feature": image_feature,
        "audio_feature": audio_feature,
        "video_feature": video_feature
    }

    return infer_data


def test():
    infer_data2 = build_infer_data_test()
    result2 = predictor.inference_item(infer_data2)
    print(infer_data2)
    print(result2)
    return infer_data2


def parse_results(results, thres=0.2):
    new_results = {}
    for label, score in results.items():
        if score > thres:
            new_results[label] = round(score, 4)
    return new_results


def setup_feature_extractor():
    text_feature_extractor = TextFeatureExtractor()
    image_feature_extractor = ImageFeatureExtractor()
    audio_feature_extractor = AudioFeatureExtractor()
    video_feature_extractor = VitVideoFeatureExtractor()
    return text_feature_extractor, image_feature_extractor, audio_feature_extractor, video_feature_extractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InsightEye Inference Script")
    parser.add_argument("--infer_config", type=str, default="configs/infer_config_e2e.yaml",
                        help="the config file to inference")
    parser.add_argument("--input_pid", type=str, default="311046937", help="the image path to inference")
    parser.add_argument("--n_gpus", type=int, default=0, help="the numbers of gpu needed")
    parser.add_argument("--best_model", type=str, default="", help="the best model for inference")
    parser.add_argument("--results_dir", type=str, default="", help="the results dir")

    args = parser.parse_args()
    config = parse_config(args.infer_config)
    config = merge_config(config, vars(args))

    tf_extractor, if_extractor, af_extractor, vf_extractor = setup_feature_extractor()
    predictor = Inferer(config)

    input_pid = args.input_pid
    inv_info = get_inv_info_by_pid(input_pid)
    # print(inv_info)

    infer_data = build_infer_data(inv_info, tf_extractor, if_extractor, af_extractor, vf_extractor,
                                  temp_dir=config.runner.data_temp_dir)
    # print(infer_data)

    results = predictor.inference_item(infer_data)
    new_results = parse_results(results)
    print(new_results)

    print("done....")
