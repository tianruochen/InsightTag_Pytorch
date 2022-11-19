#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :flask_server.py.py
# @Time     :2022/10/12 下午7:04
# @Author   :Chang Qing
 


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import json
import argparse

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import json
import math
import glob
import torch
import random
import pprint
import shutil
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

from multiprocessing import Pool
from waitress import serve
from flask import Flask, request, jsonify, make_response, render_template

from utils.comm_util import get_time_str
from utils.config_util import parse_config
from utils.server_util import log_info
from utils.server_util import error_resp

app = Flask(__name__)


@app.route('/healthcheck')
def healthcheck():
    return error_resp(0, "working")


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
        image_feature = image_feature_extractor.extract_features(image_paths=image_paths, remove_image=True)
    audio_feature = None
    video_feature = None
    if video_path:
        audio_feature = audio_feature_extractor.extract_features(video_path=video_path, temp_audios_dir=inv_dir,
                                                                 remove_audio=True)
        video_frame_dir = os.path.join(inv_dir, "video_frames")
        video_feature = video_feature_extractor.extract_features(video_path=video_path, video_frame_dir=video_frame_dir,
                                                                 remove_video=True)
    infer_data = {
        "pid": pid,
        "text_feature": text_feature,
        "image_feature": image_feature,
        "audio_feature": audio_feature,
        "video_feature": video_feature
    }

    return infer_data

def setup_feature_extractor():
    text_feature_extractor = TextFeatureExtractor()
    image_feature_extractor = ImageFeatureExtractor()
    audio_feature_extractor = AudioFeatureExtractor()
    video_feature_extractor = VitVideoFeatureExtractor()
    return text_feature_extractor, image_feature_extractor, audio_feature_extractor, video_feature_extractor


def parse_inv(inv_info):
    pid = inv_info["pid"]
    text_content = inv_info["text_content"]
    images_info = inv_info["images_info"]
    videos_info = inv_info["videos_info"]
    image_urls = []
    for image_id, image_info in images_info.items():
        image_url = image_info["img_url"]
        image_urls.append(image_url)
    video_urls = []
    for video_id, video_info in videos_info.items():
        video_url = video_info["video_url"]
        video_urls.append(video_url)

    return text_content, image_urls, video_urls

@app.route('/')
def index():
    return render_template('index.html')



@app.route("/api/inv_infer", methods=["POST"])
def inv_infer():
    if not request.method == "POST":
        return error_resp(1, "Request method error, only support [POST] method")
    start_time = str(get_time_str())
    start_tik = time.time()

    pid = request.form["pid"]
    inv_info = get_inv_info_by_pid(pid)
    # print(inv_info)

    # 解析inv_info 拿到帖子信息
    text_content, image_urls, video_urls = parse_inv(inv_info=inv_info)

    infer_data = build_infer_data(inv_info, tf_extractor, if_extractor, af_extractor, vf_extractor,
                                  temp_dir=config.runner.data_temp_dir)
    # print(infer_data)

    results = predictor.inference_item(infer_data)
    new_results = parse_results(results)

    end_time = str(get_time_str())
    end_tok = time.time()
    cost = round(end_tok - start_tik, 3)
    # write to db
    temp_dir = os.path.join(config.runner.data_temp_dir, pid)
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    res_info = {
        'start_time': start_time,
        'end_time': end_time,
        'cost_time': str(end_tok - start_tik) + 's',
        'result': new_results
    }
    res_info = str(res_info)
    print(f'#### infer {pid} done.')
    print(new_results)
    print(f"#### Infer one total cost: {cost}s ####")
    print("=" * 150)
    return render_template("index.html",
                    pid=pid,
                    text_content=text_content,
                    image_urls=image_urls,
                    video_urls=video_urls,
                    res_info=res_info)


@app.route("/api/multimodal_tag", methods=["POST"])
def multimodal_tag_infer():
    if not request.method == "POST":
        return error_resp(1, "Request method error, only support [POST] method")
    # print(request)
    # print(request.data)

    start_time = str(get_time_str())
    start_tik = time.time()

    data = json.loads(request.data)

    pid = data.get("pid")
    inv_info = get_inv_info_by_pid(pid)
    # print(inv_info)

    infer_data = build_infer_data(inv_info, tf_extractor, if_extractor, af_extractor, vf_extractor,
                                  temp_dir=config.runner.data_temp_dir)
    # print(infer_data)

    results = predictor.inference_item(infer_data)
    new_results = parse_results(results)

    end_time = str(get_time_str())
    end_tok = time.time()
    cost = round(end_tok - start_tik, 3)
    # write to db
    temp_dir = os.path.join(config.runner.data_temp_dir, pid)
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    db_info = {
        'start_time': start_time,
        'end_time': end_time,
        'cost_time': str(end_tok - start_tik) + 's',
        'result': new_results
    }

    resp = jsonify(ret=1, data=db_info)
    resp.headers['Access-Control-Allow-Origin'] = '*'

    print(f'#### infer {pid} done.')
    print(new_results)
    print(f"#### Infer one total cost: {cost}s ####")
    print("=" * 150)
    return resp


def parse_results(results, thres=0.2):
    new_results = {}
    for label, score in results.items():
        if score > thres:
            new_results[label] = round(score, 4)
    return new_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meme Classification Flask Server")
    parser.add_argument("--infer_config", type=str, default="configs/infer_config_e2e.yaml",
                        help="path of meme config(yaml file)")
    parser.add_argument("--port", type=int, default=6621, help="service port (default is 6606)")
    # parser.add_argument("--temp_dir", type=str, default="./eval_output/", help="tamp directory for post data")

    args = parser.parse_args()

    config = parse_config(args.infer_config)
    config = merge_config(config, vars(args))

    tf_extractor, if_extractor, af_extractor, vf_extractor = setup_feature_extractor()
    predictor = Inferer(config)

    if args.port:
        app.config["port"] = args.port

    serve(app, host="0.0.0.0", port=int(args.port), threads=4)
    # app.run(debug=False, host="0.0.0.0", port=int(args.port), threaded=False)

