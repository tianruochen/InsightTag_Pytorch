#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :contrastive_sort_net.py
# @Time     :2022/11/2 下午5:25
# @Author   :Chang Qing
 

"""
 多模态服务的后接模型, 主要用与对召回的标签进行排序, 聚焦主体标签
"""
from abc import ABC

import torch
import torch.nn as nn
from modules.model.bert import Bert


class ContrastiveSortNet(nn.Module):
    def __init__(self, num_classes=301):
        super(ContrastiveSortNet, self).__init__()
        self.num_classes = num_classes

        self.num_classes = num_classes
        # 降维 fc
        self.video_reduction_fc = nn.Linear(self.video_dim, self.num_classes)
        self.audio_reduction_fc = nn.Linear(self.audio_dim, self.num_classes)
        self.text_reduction_fc = nn.Linear(self.text_dim, self.num_classes)

        # bert
        self.bert = Bert(768)

        # 融合embedings维度变换
        self.combine_fc1 = nn.Linear(self.num_classes * 3 + bert_dim, 2048)
        self.combine_fc2 = nn.Linear(2048, self.num_classes)

        # labels_text embedding
        self.labels_embedding_fc1 = nn.Linear(bert_dim, 2048)
        self.labels_embedding_fc2 = nn.Linear(2048, self.num_classes)

    def forward(self, video_embedding, audio_embedding, text_embedding, recall_labels):
        # 视频特征 图片特征 文本特征 降维
        video_embedding = self.video_reduction_fc(video_embedding)
        audio_embedding = self.audio_reduction_fc(audio_embedding)
        text_embedding = self.text_reduction_fc(text_embedding)

        # 召回标签过bert
        _, recall_labels_embedding = self.recall_labels_bert(recall_labels)

        # concat 所有embedding
        combine_embedding = torch.cat((video_embedding, audio_embedding, text_embedding, recall_labels_embedding),
                                      dim=1)
        combine_embedding = self.combine_fc1(combine_embedding)
        combine_embedding = self.combine_fc2(combine_embedding)

        labels_embedding = self.labels_text_bert(self.labels_text)
        labels_embedding = self.labels_embedding_fc1(labels_embedding)
        labels_embedding = self.labels_embedding_fc2(labels_embedding)
        return combine_embedding, labels_embedding
