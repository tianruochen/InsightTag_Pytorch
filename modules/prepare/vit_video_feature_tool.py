#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :video_feature_tool.py
# @Time     :2022/6/21 上午11:40
# @Author   :Chang Qing


import os
import cv2
import timm
import math
import torch

import shutil
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

cur_script_dir = os.path.dirname(__file__)
VIT_LARGE_PATCH32_384_PATH = os.path.join(cur_script_dir, "video_extractor/weights/vit/jx_vit_large_p32_384-9b920ba8.pth")
# VIT_LARGE_PATCH32_384_PATH = '/home/work/changqing/Insight_Multimodal_Pytorch/modules/prepare/video_extractor/weights/jx_vit_large_p32_384-9b920ba8.pth'


def cutting_frame(video_path, videos_frame_root):
    video_features = None
    base_name = os.path.basename(video_path)
    base_id = base_name.split(".")[0]
    video_frame_dir = os.path.join(videos_frame_root, base_id)
    os.makedirs(video_frame_dir, exist_ok=True)

    # 切帧
    video_frame_name = os.path.join(video_frame_dir, base_id + "-%05d.jpeg")

    # 每秒提取三帧
    cmd_line = f"ffmpeg -i {video_path} -f image2 -vf fps=fps=3 -qscale:v 2 {video_frame_name} -loglevel quiet"
    # 最多提取300帧
    # cmd_line = f"ffmpeg -i {video_path} -f image2 -ss 00:00:00 -vframes 300 -qscale:v 2 {video_frame_name} -loglevel quiet"
    os.system(cmd_line)



class VitVideoFeatureExtractor:
    def __init__(self, videos_frame_root=None, batch_size=64, base_model_name="vit"):
        self.videos_frame_root = videos_frame_root  # 用来暂时存取切出来的帧
        self.batch_size = batch_size
        self.base_model_name = base_model_name
        self.model = self._build_model()
        self.config = resolve_data_config({}, model=self.model)

    def _build_model(self):
        if self.base_model_name == "vit":
            model = timm.create_model('vit_large_patch32_384', pretrained=False)
            model = model.cuda()
            checkpoint_path = VIT_LARGE_PATCH32_384_PATH
            model_data = torch.load(checkpoint_path)
            model.load_state_dict(model_data)
            model.eval()
        else:
            raise NotImplementedError
        return model

    def _build_data(self, video_frame_dir):
        frames_path = sorted(
            [os.path.join(video_frame_dir, f) for f in os.listdir(video_frame_dir) if
             os.path.isfile(os.path.join(video_frame_dir, f))])
        frame = cv2.imread(frames_path[0])
        height, width, channels = frame.shape
        num_frames = len(frames_path)

        frames = torch.FloatTensor(num_frames, channels, 384, 384)
        transform = create_transform(**self.config)

        # load the video to tensor
        for idx in range(num_frames):
            frame = Image.open(frames_path[idx]).convert('RGB')
            frame = transform(frame).unsqueeze(0)
            # print(frame.shape)
            frames[idx, :, :, :] = frame
        return frames

    def extract_features(self, video_path, video_frame_dir="", remove_video=False):
        assert (video_frame_dir or self.videos_frame_root), "video_frame_dir or slef.videos_frame_root is necessary"
        video_features = None
        base_name = os.path.basename(video_path)
        base_id = base_name.split(".")[0]
        # 优先使用函数调用时制定的dir
        if not video_frame_dir:
            video_frame_dir = os.path.join(self.videos_frame_root, base_id)
        os.makedirs(video_frame_dir, exist_ok=True)

        # 切帧
        video_frame_name = os.path.join(video_frame_dir, base_id + "-%05d.jpeg")

        # 每秒提取三帧
        # cmd_line = f"ffmpeg -i {video_path} -f image2 -vf fps=fps=3 -qscale:v 2 {video_frame_name} -loglevel quiet"
        # 最多提取300帧
        # cmd_line = f"ffmpeg -i {video_path} -f image2 -ss 00:00:00 -vframes 300 -qscale:v 2 {video_frame_name} -loglevel quiet"
        # 每秒提取三帧, 总共提取一分钟
        cmd_line = f"ffmpeg -i {video_path} -ss 00:00:00 -to 00:01:00 -f image2 -vf fps=fps=3 -qscale:v 2 {video_frame_name} -loglevel quiet"
        os.system(cmd_line)

        with torch.no_grad():
            data = self._build_data(video_frame_dir)
            if len(data.shape) > 3:
                frames_data = data.squeeze()
                if len(frames_data.shape) == 4:
                    n_chunk = len(frames_data)
                    video_features = torch.cuda.FloatTensor(n_chunk, 1024).fill_(0)
                    n_iter = int(math.ceil(n_chunk / float(self.batch_size)))
                    for i in range(n_iter):
                        min_ind = i * self.batch_size
                        max_ind = (i + 1) * self.batch_size
                        frame_batch = frames_data[min_ind:max_ind].cuda()
                        batch_features = self.model.forward_features(frame_batch)
                        video_features[min_ind:max_ind] = batch_features
                    video_features = video_features.cpu().numpy()
            else:
                print('Video {} already processed.'.format(video_path))

        # if os.path.exists(video_frame_dir):
        #     shutil.rmtree(video_frame_dir)
        if remove_video:
            os.remove(video_path)
        return video_features

    def extract_features_from_frame_dir(self, video_frame_dir):

        with torch.no_grad():
            data = self._build_data(video_frame_dir)
            if len(data.shape) > 3:
                frames_data = data.squeeze()
                if len(frames_data.shape) == 4:
                    n_chunk = len(frames_data)
                    video_features = torch.cuda.FloatTensor(n_chunk, 1024).fill_(0)
                    n_iter = int(math.ceil(n_chunk / float(self.batch_size)))
                    for i in range(n_iter):
                        min_ind = i * self.batch_size
                        max_ind = (i + 1) * self.batch_size
                        frame_batch = frames_data[min_ind:max_ind].cuda()
                        batch_features = self.model.forward_features(frame_batch)
                        print(batch_features.shape)
                        video_features[min_ind:max_ind] = batch_features
                    video_features = video_features.cpu().numpy()
            else:
                print('Video {} already processed.'.format(video_frame_dir))

        if os.path.exists(video_frame_dir):
            shutil.rmtree(video_frame_dir)

        return video_features


if __name__ == '__main__':
    extractor = VitVideoFeatureExtractor("")
    features = extractor.extract_features_from_frame_dir("/data02/tabsun/post_tag/tmp/frames/1765076885")
    print(features.shape)
