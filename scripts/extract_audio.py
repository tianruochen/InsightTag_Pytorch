#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :extract_audio.py
# @Time     :2022/7/18 下午5:48
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
from modules.prepare.audio_feature_tool import extract_audio_file
from utils.comm_util import ImageVideoCollector

if __name__ == '__main__':
    # 只用抽取audio特征
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', default=None, type=str)
    parser.add_argument('--postfix', default='mp4', type=str)
    # parser.add_argument('--frame_npy_folder', default='%s/frame_npy'%base_path, type=str)
    parser.add_argument('--audio_npy_folder', default='/data02/changqing/ZyMultiModal_Data/audios_feature', type=str)

    args = parser.parse_args()
    os.makedirs(args.audio_npy_folder, exist_ok=True)


    videos_dir = "/data02/tabsun/post_tag/val_videos/"
    # file_paths = sorted(glob.glob(videos_dir + '/*.' + args.postfix)[::-1])
    video_collector = ImageVideoCollector(collect_type="video")
    videos_path = sorted(video_collector.collect(videos_dir))

    # random.shuffle(file_paths)
    # file_paths = file_paths[:1]
    print('start extract audio features...')
    for video_file_path in tqdm.tqdm(videos_path[::-1], total=len(videos_path)):
        try:
            vid = os.path.basename(video_file_path).split('.m')[0]
            audio_npy_path = os.path.join(args.audio_npy_folder, vid + '.npy')
            if os.path.exists(audio_npy_path):
                continue
            temp_audios_dir = "/data02/changqing/ZyMultiModal_Data/temp_audios"
            base_name = os.path.basename(video_file_path).replace(".mp4", ".wav")
            audio_file = os.path.join(temp_audios_dir, base_name)
            extract_audio_file(video_file_path, audio_file=audio_file)
            # print('保存音频特征为{}'.format(audio_npy_path))
        except:
            pass


