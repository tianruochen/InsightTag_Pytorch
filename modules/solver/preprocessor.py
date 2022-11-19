#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :preprocessor.py
# @Time     :2022/10/20 上午10:42
# @Author   :Chang Qing

import os
import shutil
import traceback
from utils.zydata_util import get_inv_info_by_pid, download_file

from modules.prepare import TextFeatureExtractor, ImageFeatureExtractor
from modules.prepare import VitVideoFeatureExtractor, AudioFeatureExtractor


def parse_videos_info(videos_info, inv_dir, one_enough=True):
    videos_dir = os.path.join(inv_dir, "videos")
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


def parse_images_info(images_info, inv_dir):
    images_dir = os.path.join(inv_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    images_path = []
    for image_id, image_info in images_info.items():
        image_url = image_info["img_url"]
        image_path = download_file(str(image_id), image_url, images_dir, type="image")
        if image_path:
            images_path.append(image_path)
    return images_path


class ModalPreprocessor:
    def __init__(self, modal_dir):
        # 暂时存放多模态数据的目录
        self.modal_dir = modal_dir
        # 初始化多模态提取器
        self.text_feature_extractor = TextFeatureExtractor()
        self.image_feature_extractor = ImageFeatureExtractor()
        self.audio_feature_extractor = AudioFeatureExtractor()
        self.video_feature_extractor = VitVideoFeatureExtractor()

    def parse_modal_data(self, pid, inv_dir):
        """
        根据pid 获取帖子信息 并下载多模态数据
        :param pid: 帖子id
        :param inv_dir: 多模态数据的存放目录
        :return: 多模态数据，包括 text_content, image_paths, video_paths
        """
        inv_info = get_inv_info_by_pid(pid)
        text_content = inv_info["text_content"]
        images_info = inv_info["images_info"]
        videos_info = inv_info["videos_info"]

        # 解析图片信息，并下载图片, 返回图片地址列表
        image_paths = parse_images_info(images_info, inv_dir=inv_dir)
        # 解析视频信息，并下载视频, 返回视频地址信息
        video_path = parse_videos_info(videos_info, inv_dir=inv_dir, one_enough=True)

        return text_content, image_paths, video_path

    def extract_modal_feature(self, text_content, image_paths, video_path, inv_dir=None):
        """
        :param text_content: 文本内容
        :param image_paths: 图片路径
        :param video_path: 视频路径
        :param inv_dir: 多模态数据的存放目录，用于确定音频文件的目录
        :return: 封装好的特征信息
        """
        text_feature = self.text_feature_extractor.extract_features(text_content)
        image_feature = None
        if image_paths:
            image_feature = self.image_feature_extractor.extract_features(image_paths=image_paths, remove_image=False)
        audio_feature = None
        video_feature = None
        if video_path:
            audio_feature = self.audio_feature_extractor.extract_features(video_path=video_path,
                                                                          temp_audios_dir=inv_dir,
                                                                          remove_audio=False)
            video_frame_dir = os.path.join(inv_dir, "video_frames")
            video_feature = self.video_feature_extractor.extract_features(video_path=video_path,
                                                                          video_frame_dir=video_frame_dir,
                                                                          remove_video=False)
        modal_feature = {
            "text_feature": text_feature,
            "image_feature": image_feature,
            "audio_feature": audio_feature,
            "video_feature": video_feature
        }
        return modal_feature

    def preprocess(self, pid):
        inv_dir = os.path.join(self.modal_dir, pid)
        modal_feature = {}
        try:
            text_content, image_paths, video_path = self.parse_modal_data(pid, inv_dir=inv_dir)
            modal_feature = self.extract_modal_feature(text_content, image_paths, video_path, inv_dir=inv_dir)
        except:
            traceback.print_exc()
        finally:
            shutil.rmtree(inv_dir)
        return modal_feature
