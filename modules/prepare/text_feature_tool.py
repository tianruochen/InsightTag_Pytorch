#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :video_feature_tool.py
# @Time     :2022/6/21 上午11:40
# @Author   :Chang Qing

import os

from modules.model.pytorch_pretrained import BertTokenizer

cur_script_dir = os.path.dirname(__file__)
BERTTOKENIZER_BEST_CKPT_PATH = os.path.join(cur_script_dir, "text_extractor/weights/bert-base")
# BERTTOKENIZER_BEST_CKPT_PATH = "/home/work/changqing/Insight_Multimodal_Pytorch/modules/prepare/text_extractor/weights/bert-base"

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


class TextFeatureExtractor:

    def __init__(self, pad_size=128):
        self.pad_size = pad_size
        self.bert_ckpt_path = BERTTOKENIZER_BEST_CKPT_PATH
        self.Tokenizer = BertTokenizer.from_pretrained(self.bert_ckpt_path)

    def extract_features(self, text):
        token = self.Tokenizer.tokenize(text)
        token = [CLS] + token
        seq_len = len(token)
        token_ids = self.Tokenizer.convert_tokens_to_ids(token)

        if len(token) < self.pad_size:
            mask = [1] * len(token_ids) + [0] * (self.pad_size - len(token))
            token_ids += ([0] * (self.pad_size - len(token)))
        else:
            mask = [1] * self.pad_size
            token_ids = token_ids[:self.pad_size]
            seq_len = self.pad_size

        text_features = {
            "token_ids": token_ids,
            "seq_len": seq_len,
            "mask": mask
        }

        return text_features


if __name__ == '__main__':
    extractor = TextFeatureExtractor()
    features = extractor.extract_features("你好啊")
    print(features)
