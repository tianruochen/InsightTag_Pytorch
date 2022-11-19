#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :inferer.py
# @Time     :2022/6/21 下午5:40
# @Author   :Chang Qing


import os
import json
import time
import math
import numpy as np
import pandas as pd
import traceback

import requests
import torch
import torch.nn.functional as F

from io import BytesIO
from urllib import request
from PIL import Image
from tqdm import tqdm
from modules.solver.base import Base

from modules.loss import build_loss
from modules.metric import MultiLabel_Metric
from modules.dataset import MultiDataset, MultiModalDataset
from modules.dataset.multimodal_dataset import preprocess_frame
from torch.utils.data import DataLoader
from utils.comm_util import AverageMeter
from utils.comm_util import save_to_json, save_to_txt


def convert_datalist2dict(data_list):
    data_dict = {}
    for item in data_list:
        pid = item["pid"]
        data_dict[pid] = item
    return data_dict


def convert_datadict2list(data_dict):
    return list(data_dict.values())


def package_image_video_features(image_feature, video_feature, feature_dim=1024):
    if image_feature is None and video_feature is None:
        image_video_features = np.zeros((1, feature_dim))
        image_video_features, image_video_mask = preprocess_frame(image_video_features, 210, unsqueeze_dim=True)
        return image_video_features, image_video_mask

    # 所有图像的特征封装为 30 * self.dim 的ndarray
    image_ = np.zeros((1, feature_dim))
    if image_feature is not None:
        image_ = np.zeros((30, feature_dim))
        images_nums = image_feature.shape[0]
        seg_size = math.ceil(30 / images_nums)
        for i in range(images_nums):
            image_[i:min(i + seg_size, 30), :] = image_feature[i]

    video_ = np.zeros((0, feature_dim))
    if video_feature is not None:
        video_ = video_feature

    # 如果有视频特征没有图像特征，则将视频第一帧特征作为图像特征
    if image_ is None:
        image_ = np.zeros((30, feature_dim))
        image_[:] = video_[0]

    image_video_features = np.concatenate((image_, video_), axis=0)
    image_video_features, image_video_mask = preprocess_frame(image_video_features, 210, unsqueeze_dim=True)

    return image_video_features, image_video_mask


class Inferer(Base):
    def __init__(self, config):
        self.config = config
        super(Inferer, self).__init__(config.basic, config.arch)
        self.data = config.data
        self.runner = config.runner

        self.results_dir = self.runner.results_dir
        if not self.results_dir:
            self.results_dir = self._make_results_dir()
        self.infer_data_path = self.data.infer_data_path

        if self.infer_data_path:  # infer_data_path若存在，则batch处理
            self.batch_size = config.data.batch_size
            self.num_workers = config.data.num_workers
            self.infer_dataset = MultiModalDataset(self.infer_data_path, self.device)
            self.infer_dataloader = DataLoader(self.infer_dataset, batch_size=self.batch_size,
                                               shuffle=False)
            self.loss = build_loss(self.config.loss, self.num_classes, device=self.device)
            self.metrics = MultiLabel_Metric(self.num_classes)

        self.model.eval()

        self.analysis = config.runner.analysis
        self.save_fused_features = config.runner.save_fused_features
        if self.save_fused_features:
            self.vf_save_root = config.vf_save_root
            self.af_save_root = config.af_save_root
            self.tf_save_root = config.tf_save_root
        # self.resutls_dir = os.path.join(os.path.dirname(os.path.dirname(self.arch.best_model)),
        #                                 self.runner.results_dir_name)
        # os.makedirs(self.resutls_dir, exist_ok=True)
        # print(self.resutls_dir)

    def _make_results_dir(self):
        ckpt_path = self.arch.get("best_model", "") if self.arch.get("best_model", "") \
            else self.arch.get("resume", "")
        task_dir = os.path.dirname(os.path.dirname(ckpt_path))
        results_dir = os.path.join(task_dir, "res")
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

    def inference(self):
        test_model = self.model
        test_model.eval()
        self.metrics.reset()
        valid_pids = []
        invalid_pids = []
        predict_results = {}

        with torch.no_grad():
            # Validate
            for batch_idx, (text_, video_, audio_, label_, inv_ids) in tqdm(enumerate(self.infer_dataloader),
                                                                            total=len(self.infer_dataloader)):
                # data, target = data.to(self.device), target.to(self.device)
                output, (fused_v_f, fused_a_f, fused_t_f) = test_model(text_, video_, audio_)
                fused_v_f = fused_v_f.squeeze().cpu().numpy()
                fused_a_f = fused_a_f.squeeze().cpu().numpy()
                fused_t_f = fused_t_f.squeeze().cpu().numpy()

                inv_id = inv_ids[0]  # batch_size 只能为1的时候
                if self.save_fused_features:
                    np.save(os.path.join(self.vf_save_root, f"{inv_id}_v.npy"), fused_v_f)
                    np.save(os.path.join(self.af_save_root, f"{inv_id}_a.npy"), fused_a_f)
                    np.save(os.path.join(self.tf_save_root, f"{inv_id}_t.npy"), fused_t_f)

                have_error_data = torch.isnan(output)
                if True in have_error_data:
                    print(inv_ids[0])
                    invalid_pids.append(inv_ids[0])
                    continue
                valid_pids.extend(list(inv_ids))
                loss = self.loss(output, label_)
                self.metrics.update(output, label_, loss.item())
                batch_scores = output.cpu().tolist()
                for i, inv_id in enumerate(inv_ids):
                    name_score_dict = {}
                    inv_scores = batch_scores[i]
                    for j, score in enumerate(inv_scores):
                        name_score_dict[self.id2name[str(j)]] = round(score, 4)
                    predict_results[str(inv_id)] = name_score_dict

        predict_results_path = os.path.join(self.results_dir, "predict_results.json")
        save_to_json(predict_results, predict_results_path)
        valid_pids_path = os.path.join(self.results_dir, "valid_pids.txt")
        save_to_txt(valid_pids, valid_pids_path)
        invalid_pids_path = os.path.join(self.results_dir, "invalid_pids.txt")
        save_to_txt(invalid_pids, invalid_pids_path)
        # Record log
        # if the task_type == "multi_label", the ava_acc is top@1_acc
        # gap, avg_loss, avg_acc, avg_auc, acc_for_class, auc_for_class = self.metrics.report(top_k=5)
        infer_results = {}
        # infer_results = {
        #     'infer_gap': gap,
        #     'infer_loss': avg_loss,
        #     'infer_acc': avg_acc,
        #     "infer_auc": avg_auc,
        #     "infer_acc_for_class": acc_for_class,
        #     "infer_auc_for_class": auc_for_class
        # }
        #
        # infer_results_path = os.path.join(self.results_dir, "infer_results.json")
        # save_to_json(infer_results, infer_results_path)◊
        # print(infer_results)

        if self.analysis:
            scores_matrix_path = os.path.join(self.results_dir, "scores_matrix.npy")
            labels_matrix_path = os.path.join(self.results_dir, "labels_matrix.npy")
            scores_matrix, labels_matrix = self.metrics.export_matrix()
            np.save(scores_matrix_path, scores_matrix)
            np.save(labels_matrix_path, labels_matrix)

            # analysis_results_path = os.path.join(self.results_dir, "analysis_results.csv")
            # analysis_results = self.metrics.analysis_results(self.id2name)
            #
            # df = pd.DataFrame(analysis_results)
            # df.to_csv(analysis_results_path)
        return infer_results

    def inference_item(self, item):
        """
        :param item: = {
                "pid": pid,
                "text_feature": text_feature,
                "image_feature": image_feature,
                "audio_feature": audio_feature,
                "video_feature": video_feature
                }
        :return: name_socre_dict
        """
        text_feature = item["text_feature"]
        image_feature = item["image_feature"]
        audio_feature = item["audio_feature"]
        video_feature = item["video_feature"]

        text_ = [(torch.LongTensor(text_feature["token_ids"]).unsqueeze(0).to(self.device),
                  torch.tensor(int(text_feature["seq_len"])).unsqueeze(0).to(self.device),
                  torch.tensor(np.array(text_feature["mask"])).unsqueeze(0).to(self.device))]

        if audio_feature is None:
            audio_feature = np.zeros((1, 1024))
        audio_, audio_mask = preprocess_frame(audio_feature, 120, unsqueeze_dim=True)
        audio_ = audio_[..., :128]
        audio_ = [audio_, audio_mask]

        image_video_, image_video_mask = package_image_video_features(image_feature, video_feature)
        image_video_ = [image_video_, image_video_mask]

        test_model = self.model
        test_model.eval()
        name_score_dict = {}
        with torch.no_grad():
            # Validate
            scores, _ = test_model(text_, image_video_, audio_)
            have_error_data = torch.isnan(scores)
            if True in have_error_data:
                print("Error data...")
            scores = scores.squeeze().cpu().tolist()
            for i, score in enumerate(scores):
                name_score_dict[self.id2name[str(i)]] = score
        return name_score_dict

    def filter_nan(self, filtered_dataset_path):
        assert self.batch_size == 1, "Error! self.batch_size != 1"
        dataset_list = self.infer_dataset.data_info_list
        dataset_dict = convert_datalist2dict(dataset_list)
        test_model = self.model
        test_model.eval()
        self.metrics.reset()
        invalid_pids = []

        with torch.no_grad():
            # Validate
            for batch_idx, (text_, video_, audio_, label_, inv_ids) in tqdm(enumerate(self.infer_dataloader),
                                                                            total=len(self.infer_dataloader)):
                # data, target = data.to(self.device), target.to(self.device)
                output = test_model(text_, video_, audio_)
                have_error_data = torch.isnan(output)
                if True in have_error_data:
                    error_pid = inv_ids[0]
                    del dataset_dict[str(error_pid)]
                    self.logger.info(f"error pid {inv_ids[0]}")
                    invalid_pids.append(inv_ids[0])
                    continue
        invalid_pids_path = os.path.join(self.results_dir, "test_filtered_pids.txt")
        save_to_txt(invalid_pids, invalid_pids_path)
        filtered_dataset = convert_datadict2list(dataset_dict)
        save_to_json(filtered_dataset, filtered_dataset_path)

        ori_nums = len(dataset_list)
        new_nums = len(filtered_dataset)
        invalid_nums = len(invalid_pids)
        self.logger.info(f"ori_nums: {ori_nums}, new_nums: {new_nums}, filetered_nums: {invalid_nums}")
        return invalid_pids
