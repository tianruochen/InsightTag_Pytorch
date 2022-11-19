#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :data_statistic.py
# @Time     :2022/8/8 下午7:37
# @Author   :Chang Qing
 
import json


if __name__ == '__main__':
    """
    总帖子量： 108665
    含视频量： 44952
    含音频量： 44952
    含图片量： 106319
    含文本量： 108665
    """

    mode = "train"
    data_path = "/data02/changqing/ZyMultiModal_TestData/annotations/v1_cls301/test_cls301.json"

    """ valid:
        总帖子量： 61412
        含视频量： 35762
        含音频量： 35762
        含图片量： 60604
        含文本量： 61412
        train:
        总帖子量： 348502
        含视频量： 202512
        含音频量： 202512
        含图片量： 343841
        含文本量： 348502
    """

    data_path = "/data02/changqing/ZyMultiModal_Data/annotations/v1_cls301/train_cls301.json"
    data = json.load(open(data_path))

    v_count = 0
    a_count = 0
    i_count = 0
    t_count = 0
    only_video = 0
    only_text = 0
    only_image = 0
    for item in data:
        video_feature_paths = item["video_feature_paths"]
        audio_feature_paths = item["audio_feature_paths"]
        image_feature_paths = item["image_feature_paths"]
        text_feature_path = item["text_feature_path"]
        if video_feature_paths:
            v_count += 1
        if audio_feature_paths:
            a_count += 1
        if image_feature_paths:
            i_count += 1
        if text_feature_path:
            t_count += 1

        if not text_feature_path and not image_feature_paths and video_feature_paths:
            only_video += 1
        if not video_feature_paths and not image_feature_paths and text_feature_path:
            only_text += 1
        if not video_feature_paths and not text_feature_path and image_feature_paths:
            only_image += 1
    print(f"总帖子量： {len(data)}")
    print(f"含视频量： {v_count}")
    print(f"含音频量： {a_count}")
    print(f"含图片量： {i_count}")
    print(f"含文本量： {t_count}")
    print(f"纯视频量： {only_video}")
    print(f"含图片量： {only_image}")
    print(f"纯文本量： {only_text}")





