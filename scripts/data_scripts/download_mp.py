#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :download_imgs_mp.py
# @Time     :2021/10/27 下午3:29
# @Author   :Chang Qing


import os
import json
import time
import random
import requests
import argparse
import traceback

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

requests.DEFAULT_RETRIES = 5
s = requests.session()
s.keep_alive = False
random.seed(666)


def download(item):
    # name, url = item
    base_name, url = item
    data_path = os.path.join(data_root, base_name)
    if not os.path.exists(data_path):
        try:
            res = requests.get(url, timeout=1)
            if res.status_code != 200:
                raise Exception
            with open(data_path, "wb") as f:
                f.write(res.content)
        except Exception as e:
            print(base_name, url)
            traceback.print_exc()


def build_url_list(url_file):
    url_list = []
    with open(url_file) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            url_list.append([str(i), line.strip()])
    return url_list


def build_items(item_file):
    items = []
    with open(item_file) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or len(line.split("\t")) != 3:
                continue
            pid, img_id, url = line.split("\t")

            items.append(f"{pid}_{img_id}\t{url}")
    return items


def build_item_list(data, type):
    item_list = list()

    for inv_id, inv_info in data.items():
        if type == "image":
            images_info = inv_info["images_info"]
            for image_id, image_info in images_info.items():
                base_name = f"{inv_id}_{image_id}.jpg"
                image_url = image_info["img_url"]
                item_list.append((base_name, image_url))
        elif type == "video":
            videos_info = inv_info["videos_info"]
            for video_id, video_info in videos_info.items():
                base_name = f"{inv_id}_{video_id}.mp4"
                video_url = video_info["video_url"]
                item_list.append((base_name, video_url))
    return item_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Images Download Script")
    parser.add_argument("--root", default="/data1/changqing/ZyMultiModal_Data/videos",
                        type=str,
                        help="the directory of images")
    parser.add_argument("--type", default="video", type=str, help="the file type to download")
    parser.add_argument("--workers", default=3, type=int, help="the nums of process")
    args = parser.parse_args()

    data_root = args.root
    type = args.type
    workers = args.workers

    cls24_data_path = "/data1/changqing/ZyMultiModal_Data/annotations/v1_cls24/filtered_invs_data_20210101_20220430_cls24.json"
    other_data_path = "/data1/changqing/ZyMultiModal_Data/annotations/v1_cls24/filtered_invs_data_20220101_20220430_cls24_others.json"

    cls24_data = json.load(open(cls24_data_path))
    other_data = json.load(open(other_data_path))

    cls24_data.update(other_data)
    data = cls24_data

    item_list = build_item_list(data, type)

    tik_time = time.time()
    # create multiprocess pool
    pool = Pool(workers)  # process num: 20

    # 如果check_img函数仅有1个参数，用map方法
    # pool.map(check_img, img_paths)
    # 如果check_img函数有不止1个参数，用apply_async方法
    # for img_path in tqdm(img_paths):
    #     pool.apply_async(check_img, (img_path, False))
    list(tqdm(iterable=(pool.imap(download, item_list)), total=len(item_list)))
    pool.close()
    pool.join()
    tok_time = time.time()
    print(tok_time - tik_time)


    # url_txt_list = sorted(glob("data/drawing_board/test/*.txt"))
    # print(url_txt_list)
    # imgs_root = ""
    # for url_txt in url_txt_list:
    #     date_str = os.path.basename(url_txt)[:-4].split("_")[-1]
    #     imgs_root = f"/Users/zuiyou/PycharmProjects/Image_Process_Tool/images/{date_str}"
    #     workers = args.workers
    #
    #     os.makedirs(imgs_root, exist_ok=True)
    #
    #     # url_file = "imgs.txt"
    #     # url_file = "others_20211214-20211223.txt"
    #     # items = open(url_file).readlines()
    #     items = build_items(url_txt)
    #     # other类太多，分批处理， 一次处理20000张
    #     items = [item.strip() for item in items if item]
    #     # url_list = build_url_list(url_file)
    #     print(f"date_str: {date_str}   total items: {len(items)}")
    #
    #     # random.shuffle(url_list)
    #     # url_list = url_list[:180000]
    #
    #     tik_time = time.time()
    #     # create multiprocess pool
    #     pool = Pool(workers)  # process num: 20
    #
    #     # 如果check_img函数仅有1个参数，用map方法
    #     # pool.map(check_img, img_paths)
    #     # 如果check_img函数有不止1个参数，用apply_async方法
    #     # for img_path in tqdm(img_paths):
    #     #     pool.apply_async(check_img, (img_path, False))
    #     list(tqdm(iterable=(pool.imap(download_img, items)), total=len(items)))
    #     pool.close()
    #     pool.join()
    #     tok_time = time.time()
    #     print(tok_time - tik_time)
