#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :extract_text_features.py
# @Time     :2022/6/6 下午2:50
# @Author   :Chang Qing

import os
import sys
sys.path.append("..")
import json
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from modules.prepare.text_feature_tool import TextFeatureExtractor


def build_ocr_dict():
    ocr_dict = defaultdict(str)
    ocr_data = open("../data/total_dataset_img_ocr.txt", "r").readlines()
    for line in ocr_data:
        line = line.strip()
        if len(line.split("\t")) == 3:
            inv_id, image_id, ocr_content = line.split("\t")
            ocr_dict[inv_id] += ocr_content + "。"
    return ocr_dict



if __name__ == '__main__':
    texts_feature_root = "/data02/changqing/ZyMultiModal_Data/ocrs_feature"
    # raw_dir = "/data02/tabsun/post_tag/raw"
    # category_files = glob(os.path.join(raw_dir, "*.json"))
    ocr_dict = build_ocr_dict()

    text_feature_extractor = TextFeatureExtractor()

    for inv_id, ocr_content in tqdm(ocr_dict.items(), total=len(ocr_dict)):
        save_path = os.path.join(texts_feature_root, f"{inv_id}.json")
        if os.path.exists(save_path):
            continue
        text_content = ocr_content.strip()
        content_feature = text_feature_extractor.extract_features(text_content)
        text_feature = {
            "content_feature": content_feature,
            "topic_feature": {}
        }
        json.dump(text_feature, open(save_path, "w"), ensure_ascii=False, indent=4)
        # break
#
# if __name__ == '__main__':
#     texts_feature_root = "/data02/changqing/ZyMultiModal_Data/ocr_contents_feature"
#     # raw_dir = "/data02/tabsun/post_tag/raw"
#     # category_files = glob(os.path.join(raw_dir, "*.json"))
#     ocr_dict = build_ocr_dict()
#
#     raw_data_path = "/data02/tabsun/post_tag/support_samples.json"
#     category_files = [raw_data_path]
#
#     text_feature_extractor = TextFeatureExtractor()
#
#     for category_file in category_files:
#         print(os.path.basename(category_file).replace("__", "_").split(".")[0])
#         category_raw_data = json.load(open(category_file))
#
#         for inv_info in tqdm(category_raw_data):
#             inv_id = inv_info["pid"]
#             save_path = os.path.join(texts_feature_root, f"{inv_id}.json")
#             if os.path.exists(save_path):
#                 continue
#             text_content = inv_info["text_content"].strip()
#             content_feature = text_feature_extractor.extract_features(text_content)
#             text_feature = {
#                 "content_feature": content_feature,
#                 "topic_feature": {}
#             }
#             json.dump(text_feature, open(save_path, "w"), ensure_ascii=False, indent=4)


    # data = json.load(open("/data1/changqing/ZyMultiModal_Data/annotations/v1_cls24/total_invs_data_20210101_20220430_cls24.json"))
    #
    # for inv_id, inv_info in data.items():
    #     text_content = inv_info["text_content"].strip()
    #     topic_name = inv_info["topic_name"].strip()
    #     print(text_content)
    #     print(topic_name)
    #     content_feature = text_feature_extractor.extract_features(text_content)
    #     topic_feature = text_feature_extractor.extract_features(topic_name)
    #     text_feature = {
    #         "content_feature": content_feature,
    #         "topic_feature": topic_feature
    #     }
    #     save_path = os.path.join(texts_feature_root, f"{inv_id}.json")
    #     json.dump(text_feature, open(save_path, "w"), ensure_ascii=False, indent=4)
    # content_feature = text_feature_extractor.extract_features("")
    # # topic_feature = TextFeatureExtractor.extract_features("")
    # text_feature = {
    #         "content_feature": content_feature,
    #         "topic_feature": {}
    #     }
    # print(text_feature)
    # save_path = os.path.join(texts_feature_root, "no_data.json")
    # json.dump(text_feature, open(save_path, "w"), ensure_ascii=False, indent=4)

    # id2text_content_path = "/data02/changqing/ZyMultiModal_Data/annotations/v1_cls301/id2text_content_new.json"
    # id2text_content = dict()
    # category_files += ["/data02/tabsun/post_tag/test_samples.json"]
    # for category_file in category_files:
    #     category_raw_data = json.load(open(category_file))
    #
    #     for inv_info in tqdm(category_raw_data):
    #         inv_id = inv_info["pid"]
    #         text_content = inv_info["text_content"].strip()
    #         id2text_content[inv_id] = text_content
    #
    # json.dump(id2text_content, open(id2text_content_path, "w"), ensure_ascii=False, indent=4)