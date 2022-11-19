#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :multi_label_metric.py
# @Time     :2022/6/21 下午7:53
# @Author   :Chang Qing

import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from modules.metric.gap_cal import calculate_gap
from utils.analysis_util import metrix_analysis

class MultiLabel_Metric:
    def __init__(self, cls_nums):
        self.eps = 1e-7
        self.cls_nums = cls_nums
        self.reset()
        self.zero_tensor = torch.zeros(cls_nums)

    def update(self, batch_pred, batch_label, batch_loss):
        """
        :param pred: used sigmoid (bs, cls_nums)
        :param label:  one_hot (bs, cls_nums)
        :param loss: scalar
        :return:
        """
        batch_count = batch_pred.shape[0]
        self.img_nums += batch_count
        self.lose_sum += batch_loss
        # batch_pred = F.sigmoid(batch_pred)
        # batch_label = batch_label.cpu().tolist()
        self.pd_scores.extend(batch_pred.cpu().tolist())
        self.gt_labels.extend(batch_label.cpu().tolist())
        top1_batch_matched = 0

        for i in range(batch_count):
            # temp_one_hot = [0] * self.cls_nums
            # temp_one_hot[label[i]] = 1
            # self.total_num_for_class += label
            if (torch.equal(batch_label[i].cpu(), self.zero_tensor) and float(torch.max(batch_label[i])) < 0.4) or \
                    batch_label[i][torch.argmax(batch_pred[i])] == 1:
                # self.right_num_for_class[label[i]] += 1
                top1_batch_matched += 1
        self.top1_total_matched += top1_batch_matched
        top1_batch_accuracy = top1_batch_matched / (batch_count + self.eps)
        return top1_batch_accuracy, top1_batch_matched

    def reset(self):
        self.img_nums = 0
        self.lose_sum = 0
        self.top1_total_matched = 0
        self.gt_labels = []
        self.pd_scores = []
        self.error_pd = {}
        # self.right_num_for_class = [0] * self.cls_nums
        self.total_num_for_class = [0] * self.cls_nums

    def report(self, top_k=2):
        avg_loss = self.lose_sum / (self.img_nums + self.eps)
        top1_acc = self.top1_total_matched / (self.img_nums + self.eps)

        # 不指定dtype，默认是float64, 过大的数据量会撑爆内存
        gt_labels = np.array(self.gt_labels, dtype=np.float16)
        pd_scores = np.array(self.pd_scores, dtype=np.float16)
        gt_labels[gt_labels < 1] = 0
        # print(gt_labels[:, 0])
        # print(pd_scores[:, 0])
        # auc_for_class = [roc_auc_score(gt_labels[:, i], pd_scores[:, i]) for i in range(self.cls_nums)]
        # avg_auc = sum(auc_for_class) / self.cls_nums
        # acc_for_class = None
        avg_auc = None
        acc_for_class = None
        auc_for_class = None
        gap = calculate_gap(pd_scores, gt_labels, top_k=top_k)

        return gap, avg_loss, top1_acc, avg_auc, acc_for_class, auc_for_class

    def export_matrix(self):
        scores_matrix = np.array(self.pd_scores, dtype=np.float16)
        labels_matrix = np.array(self.gt_labels, dtype=np.float16)
        return scores_matrix, labels_matrix

    def analysis_results(self, id2name):
        """
        分析结果，计算pr值
        :param idx2name: 类别索引到类别名称的映射，json
        :return: res
        """
        scores_matrix = np.array(self.pd_scores, dtype=np.float16)
        labels_matrix = np.array(self.gt_labels, dtype=np.float16)
        results = metrix_analysis(scores_matrix, labels_matrix, id2name)
        return results

