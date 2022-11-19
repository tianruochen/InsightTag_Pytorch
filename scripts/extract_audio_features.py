#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :extract_audio_features.py
# @Time     :2022/6/21 上午11:23
# @Author   :Chang Qing
 

import sys, os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

sys.path.append(os.getcwd())

import time
import argparse
import tqdm
import random
import glob
import traceback
sys.path.append("..")
import numpy as np
from modules.prepare.audio_feature_tool import AudioFeatureExtractor
from utils.comm_util import ImageVideoCollector

if __name__ == '__main__':
    # 只用抽取audio特征
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', default=None, type=str)
    parser.add_argument('--postfix', default='mp4', type=str)
    # parser.add_argument('--frame_npy_folder', default='%s/frame_npy'%base_path, type=str)
    parser.add_argument('--audio_npy_folder', default='/data02/changqing/ZyMultiModal_SuppData/audios_feature', type=str)

    args = parser.parse_args()
    os.makedirs(args.audio_npy_folder, exist_ok=True)

    audio_feature_extractor = AudioFeatureExtractor()

    videos_dir = "/data02/tabsun/post_tag/support_videos"
    # file_paths = sorted(glob.glob(videos_dir + '/*.' + args.postfix)[::-1])
    video_collector = ImageVideoCollector(collect_type="video")
    # videos_path = sorted(video_collector.collect(videos_dir))[::-1]
    videos_path = video_collector.collect(videos_dir)
    random.shuffle(videos_path)
    videos_nums = len(videos_path)
    # start = 40943
    # duration = 1020
    # videos_path = videos_path[start:start+duration][::-1]
    # print(len(videos_path))
    # random.shuffle(file_paths)
    # file_paths = file_paths[:1]
    print('start extract audio features...')
    while True:
        for video_file_path in tqdm.tqdm(videos_path, total=len(videos_path)):
            try:
                vid = os.path.basename(video_file_path).split('.m')[0]
                audio_npy_path = os.path.join(args.audio_npy_folder, vid + '.npy')
                if os.path.exists(audio_npy_path):
                    continue
                audio_features = audio_feature_extractor.extract_features(video_file_path)
                np.save(audio_npy_path, audio_features)
                # print('保存音频特征为{}'.format(audio_npy_path))
            except:
                pass
        time.sleep(10)
        videos_path = video_collector.collect(videos_dir)
        random.shuffle(videos_path)
        new_videos_nums = len(videos_path)
        if new_videos_nums == videos_nums:
            exit(0)
        else:
            videos_nums = new_videos_nums


