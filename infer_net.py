#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :train_net.py.py
# @Time     :2022/6/21 下午5:41
# @Author   :Chang Qing
 


import json
import os
import cv2
import glob
import random
import pprint
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tqdm import tqdm
from utils.config_util import parse_config
from utils.config_util import merge_config
from modules.solver.inferer import Inferer
from utils.comm_util import save_to_json, save_to_txt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InsightEye Inference Script")
    parser.add_argument("--infer_config", type=str, default="configs/infer_config_v1_cls301.yaml",
                        help="the config file to inference")
    parser.add_argument("--data_path", type=str, default="", help="the image path to inference")
    parser.add_argument("--n_gpus", type=int, default=0, help="the numbers of gpu needed")
    parser.add_argument("--best_model", type=str, default="", help="the best model for inference")
    parser.add_argument("--results_dir", type=str, default="", help="the results dir")

    args = parser.parse_args()

    config = parse_config(args.infer_config)
    config = merge_config(config, vars(args))

    predictor = Inferer(config)
    result = predictor.inference()
    # predictor.filter_nan("/data02/changqing/ZyMultiModal_Data/annotations/v2_cls301/train_cls301_86w_filtered.json")
    print("done....")

