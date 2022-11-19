#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :tsn_video_feature_tool.py
# @Time     :2022/7/8 下午5:42
# @Author   :Chang Qing


import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["FLAGS_eager_delete_tensor_gb"] = "0"
# os.environ["FLAGS_memory_fraction_of_eager_deletion"] = "1"
# os.environ["FLAGS_fast_eager_deletion_mode"] = "True"
import sys
sys.path.append("../../")
import time
import logging
import argparse
import ast
import warnings

warnings.filterwarnings("ignore")

import traceback
import numpy as np
import paddle
import paddle.fluid as fluid

from utils.config_util import *
from utils.utility import check_cuda
from utils.utility import check_version
from utils.comm_util import ImageVideoCollector
from modules.prepare.video_extractor import models
from modules.prepare.video_extractor import ZYVideoReader
from tqdm import tqdm

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--extractor_config',
        type=str,
        default='configs/tsn.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--extractor_name',
        type=str,
        default='TSN',
        help='extractor model name, default TSN')
    parser.add_argument(
        '--predictor_config',
        '--pconfig',
        type=str,
        default='configs/attention_lstm.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--predictor_name',
        '--pname',
        type=str,
        default='AttentionLSTM',
        help='predictor model name, as AttentionLSTM, AttentionCluster, NEXTVLAD'
    )
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--extractor_weights',
        type=str,
        default='weights/tsn',
        help='extractor weight path')
    parser.add_argument(
        '--predictor_weights',
        '--pweights',
        type=str,
        default='weights/attention_lstm',
        help='predictor weight path')
    parser.add_argument(
        '--filelist',
        type=str,
        default='./data/VideoTag_test.list',
        help='path of video data, multiple video')
    parser.add_argument(
        '--video_path',
        type=str,
        default="./data/mp4/5.mp4"
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='data/VideoTag_results',
        help='output file path')
    parser.add_argument(
        '--label_file',
        type=str,
        default='label_3396.txt',
        help='chinese label file path')

    args = parser.parse_args()
    return args


class TSN_Extractor:

    def __init__(self, extractor_config="configs/tsn.yaml", extractor_weights="weights/tsn"):
        self.extractor_infer_config = parse_config(extractor_config)
        logging.info(self.extractor_infer_config)
        extractor_start_time = time.time()
        self.extractor_scope = fluid.Scope()
        # paddle 2.0rc 默认是动态图  因此需要开启静态图模式
        paddle.enable_static()
        with fluid.scope_guard(self.extractor_scope):
            self.extractor_startup_prog = fluid.Program()
            self.extractor_main_prog = fluid.Program()
            with fluid.program_guard(self.extractor_main_prog, self.extractor_startup_prog):
                with fluid.unique_name.guard():
                    # build model
                    self.extractor_model = models.get_model(
                        "TSN",
                        self.extractor_infer_config,
                        mode='infer',
                        is_videotag=True)
                    # input_shape = [None, 300, 3 ,target_size, target_size]
                    self.extractor_model.build_input(use_dataloader=False)
                    # model_output : (None, 300, 2048)
                    self.extractor_model.build_model()
                    # get model input : self.feature_input = [image]
                    # input_shape = [None, 300, 3 ,target_size, target_size]
                    self.extractor_feeds = self.extractor_model.feeds()
                    # get model output : self.network_outputs
                    # model_output : (None, 300, 2048)
                    self.extractor_fetch_list = self.extractor_model.fetches()

                    self.place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
                    self.exe = fluid.Executor(self.place)

                    self.exe.run(self.extractor_startup_prog)

                    logger.info('load extractor weights from {}'.format(
                        args.extractor_weights))
                    load_extractor_tik = time.time()
                    self.extractor_model.load_pretrain_params(
                        self.exe, extractor_weights, self.extractor_main_prog)
                    load_extractor_tok = time.time()
                    logger.info("======load extractor weight cost : {}".format(load_extractor_tok - load_extractor_tik))

    def extract(self, filepath):
        tsn_extractor_reader = ZYVideoReader("ZYVideo", 'infer', self.extractor_infer_config)
        with fluid.scope_guard(self.extractor_scope):
            images = tsn_extractor_reader.decode(filepath)
            tsn_extractor_outputs = self.exe.run(self.extractor_main_prog, fetch_list=self.extractor_fetch_list,
                                                 feed={"image": [images]})
            return tsn_extractor_outputs[0]

    def extract_from_imgs(self, imgs):
        with fluid.scope_guard(self.extractor_scope):
            tsn_extractor_outputs = self.exe.run(self.extractor_main_prog, fetch_list=self.extractor_fetch_list,
                                                 feed={"image": [imgs]})
            return tsn_extractor_outputs[0]


class LSTM_Predictor:
    def __init__(self, predictor_config="configs/attention_lstm.yaml", predictor_weights="weights/attention_lstm"):
        self.predictor_infer_config = parse_config(predictor_config)
        predictor_start_time = time.time()
        self.predictor_scope = fluid.Scope()
        with fluid.scope_guard(self.predictor_scope):
            self.predictor_startup_prog = fluid.Program()
            self.predictor_main_prog = fluid.Program()
            with fluid.program_guard(self.predictor_main_prog, self.predictor_startup_prog):
                with fluid.unique_name.guard():
                    # parse config
                    self.predictor_model = models.get_model(
                        "AttentionLSTM", self.predictor_infer_config, mode='infer')
                    self.predictor_model.build_input(use_dataloader=False)
                    self.predictor_model.build_model()
                    self.predictor_feeds = self.predictor_model.feeds()
                    self.predictor_fetch_list = self.predictor_model.fetches()

                    self.place = fluid.CUDAPlace(0)
                    self.exe = fluid.Executor(self.place)
                    self.exe.run(self.predictor_startup_prog)

                    logger.info('load predictor weights from {}'.format(predictor_weights))

                    load_predictor_tik = time.time()
                    self.predictor_model.load_test_weights(self.exe, predictor_weights,
                                                           self.predictor_main_prog)
                    load_predictor_tok = time.time()
                    logging.info(
                        "======load predictor weight cost : {}".format(load_predictor_tok - load_predictor_tik))

    def predict(self, tsn_out_features):
        # get Predictor input from Extractor output
        predictor_feeder = fluid.DataFeeder(place=self.place, feed_list=self.predictor_feeds)
        with fluid.scope_guard(self.predictor_scope):
            predictor_outputs = self.exe.run(self.predictor_main_prog, fetch_list=self.predictor_fetch_list,
                                             feed=predictor_feeder.feed([tsn_out_features]))
            return predictor_outputs


class TsnVideoFeatureExtractor:
    def __init__(self):
        extractor_config = "video_extractor/configs/tsn.yaml"
        extractor_weights = "video_extractor/weights/tsn.pdparams"
        self.tsn_extractor = TSN_Extractor(extractor_config=extractor_config, extractor_weights=extractor_weights)

        predictor_config = "video_extractor/configs/attention_lstm.yaml"
        predictor_weights = "video_extractor/weights/attention_lstm.pdparams"
        self.lstm_predictor = LSTM_Predictor(predictor_config=predictor_config, predictor_weights=predictor_weights)

    def extract_features(self, video_path):
        tsn_feature = self.tsn_extractor.extract(video_path)
        lstm_feature = self.lstm_predictor.predict(tsn_feature)[1]
        return tsn_feature, lstm_feature


if __name__ == '__main__':
    args = parse_args()
    # pid_url_dict = parse_csv("finetune_zy/zy_data/video_post_label_with_label_11w.csv")
    check_cuda(args.use_gpu)
    check_version()
    logger.info(args)

    videos_root = "/data02/tabsun/post_tag/val_videos/"
    tsn_features_root = "/data02/changqing/ZyMultiModal_Data/videos_feature/tsn_lstm/tsn1x300x2048"
    lstm_features_root = "/data02/changqing/ZyMultiModal_Data/videos_feature/tsn_lstm/lstm1x4096"

    video_collector = ImageVideoCollector(collect_type="video")
    videos_path = video_collector.collect(videos_root)

    tsn_video_feature_extractor = TsnVideoFeatureExtractor()
    invalid_videos_path = open("invalid_videos_path.txt").readlines()
    invalid_videos_path = [invalid_video_path.strip() for invalid_video_path in invalid_videos_path if invalid_video_path]
    for video_path in tqdm(videos_path):
        if video_path in invalid_videos_path:
            continue
        try:
            logging.info(video_path)
            basename = os.path.basename(video_path)[:-4]
            feature_base_name = basename + ".npy"
            tsn_feature_path = os.path.join(tsn_features_root, feature_base_name)
            lstm_feature_path = os.path.join(lstm_features_root, feature_base_name)
            if os.path.exists(tsn_feature_path) and os.path.exists(lstm_feature_path):
                continue

            tsn_feature, lstm_feature = tsn_video_feature_extractor.extract_features(video_path)

            # logging.info(tsn_feature.shape)
            # logging.info(lstm_feature.shape)
            np.save(tsn_feature_path, tsn_feature)
            np.save(lstm_feature_path, lstm_feature)
        except:
            traceback.print_exc()
            continue
