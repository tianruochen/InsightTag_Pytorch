#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :comm_util.py
# @Time     :2021/3/26 上午11:18
# @Author   :Chang Qing

import os
import glob
import time
import json
import yaml
import logging
import datetime
import logging.config

import torch

from collections import OrderedDict


class ResultsLog:

    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def is_image(file_path):
    ext_list = [".png", ".jpg", ".jpeg", ".gif"]
    file_ext = os.path.splitext(file_path)[-1].lower()
    return file_ext in ext_list


def is_video(file_path):
    ext_list = [".mp4"]
    file_ext = os.path.splitext(file_path)[-1].lower()
    return file_ext in ext_list


class ImageVideoCollector(object):
    """Class for collecting pictures"""

    def __init__(self, collect_type="image"):
        self.collect_type = collect_type

    def collect_dirs(self, root_dir, recursive=False, abs_path=False):
        dir_list = []
        for root, dirs, files in os.walk(root_dir):
            for dir_name in dirs:
                if abs_path:
                    dir = os.path.join(root, dir_name)
                dir_list.append(dir_name)
                if recursive:
                    img_list.extend(self.collect_dirs(os.path.join(root, dir_name)))
        return dir_list

    def collect(self, root_dir):
        path_list = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if (self.collect_type == "image" and is_image(file_path)) or \
                        (self.collect_type == "video" and is_video(file_path)):
                    path_list.append(file_path)
                    # if os.path.islink(file_path):
                    #     symbolic_link_point = os.readlink(file_path)
                    #     if os.path.isdir(symbolic_link_point):
                    #         img_list.extend(self.collect_imgs(symbolic_link_point))
        return list(set(path_list))

    def collect_imgs(self, root_dir):
        img_list = []
        for root, dirs, files in os.walk(root_dir):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                img_list.extend(self.collect_imgs(dir_path))
            for file in files:
                file_path = os.path.join(root, file)
                if is_image(file_path):
                    img_list.append(file_path)
                # if os.path.islink(file_path):
                #     symbolic_link_point = os.readlink(file_path)
                #     if os.path.isdir(symbolic_link_point):
                #         img_list.extend(self.collect_imgs(symbolic_link_point))
        return list(set(img_list))

    def collect_imgs2(self, root_dir):
        img_list = []
        dirs = os.listdir(root_dir)
        for dir_name in dirs:
            dir_path = os.path.join(root_dir, dir_name)
            one_list = glob.glob(os.path.join(dir_path, "*.jpg")) + glob.glob(os.path.join(dir_path, "*.jpeg")) + \
                       glob.glob(os.path.join(dir_path, "*.png")) + glob.glob(os.path.join(dir_path, "*.gif"))
            img_list.extend(one_list)
        return img_list


def setup_logger(default_path=None, default_level=logging.INFO):
    """
    Set up logging configuration
    :param default_path: file to logging configuration
    :param default_level: logging level (default: logging.INFO)
    :return: root logger
    """
    if default_path and os.path.isfile(default_path):
        with open(default_path, "rt") as f:
            logger_conf = yaml.load(f, Loader=yaml.Loader)
        logging.config.dictConfig(logger_conf)
    else:
        logging.basicConfig(level=default_level, format="[%(asctime)15s][%(levelname)6s][%(filename)s]: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    return logging.getLogger("root")


def setup_device(n_gpu_need):
    """
    check training gpu environment
    :param n_gpu_need: int
    :return:
    """
    logger = logging.getLogger("root")
    n_gpu_available = torch.cuda.device_count()
    gpu_list_ids = []
    if n_gpu_need == 0:
        logger.info("run models on CPU.")
    if n_gpu_available == 0 and n_gpu_need > 0:
        logger.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
    elif n_gpu_need > n_gpu_available:
        n_gpu_need = n_gpu_available
        logger.warining(
            "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                n_gpu_need, n_gpu_available))
    else:
        logging.info(f"run model on {n_gpu_need} gpu(s)")
        gpu_list_str = os.environ["CUDA_VISIBLE_DEVICES"]
        gpu_list_ids = [int(i) for i in gpu_list_str.split(",")][:n_gpu_need]
    return n_gpu_need, gpu_list_ids


def get_time_str():
    timestamp = time.time()
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    return time_str


def save_to_txt(item_list, save_path):
    item_list = [item + "\n" if not item.endswith("\n") else item for item in item_list]
    # print(item_list)
    with open(save_path, "w") as f:
        f.writelines(item_list)


def save_to_json(save_target, save_path):
    json.dump(save_target, fp=open(save_path, "w"), indent=4, ensure_ascii=False)


def get_date_str(n_days_ago=0):
    focus_day = datetime.datetime.today().date() - datetime.timedelta(days=n_days_ago)
    focus_day_str = time.strftime("%Y%m%d", time.strptime(str(focus_day), '%Y-%m-%d'))
    # focus_day_str = time.mktime(time.strptime(str(focus_day), '%Y-%m-%d'))
    return focus_day_str


def sort_dict(ori_dict, by_key=False, reverse=False):
    """
    sorted dict by key or value
    :param ori_dict:
    :param by_key: sorted by key or value
    :param reverse: if reverse is true, big to small. if false, small to big
    :return: OrderedDict
    """
    ordered_list = sorted(ori_dict.items(), key=lambda item: item[0] if by_key else item[1])
    ordered_list = ordered_list[::-1] if reverse else ordered_list
    new_dict = OrderedDict(ordered_list)
    return new_dict


def reverse_dict(ori_dict):
    new_dict = dict()
    try:
        for key, value in ori_dict.items():
            new_dict[str(value)] = key
    except:
        print("reverse error")
        return ori_dict
    return new_dict


def build_mapping_from_list(name_list):
    """
    :param name_list:
    :return: idx2name_map, name2idx_map
    """
    name_list = sorted([name for name in list(set(name_list)) if name])
    idx2name_map = OrderedDict()
    name2idx_map = OrderedDict()
    for idx, name in enumerate(name_list):
        idx = str(idx)
        name = str(name)
        idx2name_map[idx] = name
        name2idx_map[name] = idx
    return idx2name_map, name2idx_map


if __name__ == "__main__":
    # a = ["1", "2" + "\n", "3", "4" + "\n", "5"]
    # # save_to_txt(a, "test.txt")
    #
    # ori_dict = {
    #     "a": 3,
    #     "c": 1,
    #     "b": 2,
    # }
    # print(ori_dict)
    # print(sort_dict(ori_dict, by_key=False, reverse=False))
    # print(sort_dict(ori_dict, by_key=False, reverse=True))
    # print(sort_dict(ori_dict, by_key=True, reverse=False))
    # print(sort_dict(ori_dict, by_key=True, reverse=True))
    root_dir = "/data1/changqing/ZyImage_Data/images_mulit_label"
    dir_list = ImageVideoCollector().collect_dirs(root_dir)
    print(dir_list)
    print(len(dir_list))

    img_list = ImageVideoCollector().collect_imgs2(root_dir)
    print(len(img_list))
    print(len(list(set(img_list))))
    save_to_txt(img_list, "/data1/changqing/ZyImage_Data/annotations/imgtag_v04/imgtag_v04_multi_label_images.txt")
