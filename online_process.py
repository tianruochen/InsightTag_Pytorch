#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :online_process.py
# @Time     :2022/10/19 下午4:14
# @Author   :Chang Qing


import argparse
import cv2
import sys
import time
from threading import Thread, Lock

from king import King
from utils.db_util import get_batch_items
from utils.collection_util import MongoCollection
from utils.config_util import parse_config, merge_config

if sys.version_info > (3, 0):
    import queue as Queue
else:
    import Queue


def time_step(prev_t):
    return int((cv2.getTickCount() - prev_t) / cv2.getTickFrequency())


def King_on_the_way(thread_id):
    global King_pool
    global database
    global task_queue

    count = 0
    while not task_queue.empty():
        print("Thread %d still working" % thread_id)
        count += 1
        mutex.acquire()
        if task_queue.empty():
            mutex.release()
            break
        else:
            sample = task_queue.get()
        mutex.release()
        # process
        # print(f"!!!第{thread_id}个线程开始工作...")
        King_pool[thread_id].kill(sample)


##################################
#  main
##################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VideoTag Model Forward Part.')
    parser.add_argument('--config', type=str, default='./configs/multimodal_online_infer_pipe.yaml',
                        help='Config file for specific project')
    parser.add_argument('--parallel_num', type=int, default=1, help='Multi thread num for process')
    parser.add_argument('--query_time_step', type=int, default=10,
                        help='How often to query the db for new samples to be processed (in seconds)')
    # parser.add_argument('--pid', type=str, default='pid.txt',
    #                    help='The file to record pid')
    args = parser.parse_args()
    # with open(args.config, 'r') as f:
    #     params = json.load(f)
    config = parse_config(args.config)
    config = merge_config(config, vars(args))

    # with open(args.pid, 'w') as f:
    #    f.write(str(os.getpid()))

    # Call N thread with each thread own one removor model
    parallel_thread_num = config.runner.parallel_num
    King_pool = []
    for i in range(parallel_thread_num):
        king = King(config, king_id=i)
        King_pool.append(king)
    print("*******THE GUN IS LOADED*********")

    db_src_params = config.database['db_src_params'].copy()
    src_conditions = [["processed:exist:False"], ["processed:=:False"]]
    src_collection = MongoCollection(db_params=db_src_params)
    # Task queue to process
    task_queue = Queue.Queue()
    mutex = Lock()
    query_time_step = config.runner.query_time_step

    batch_num = 0
    prev_t = cv2.getTickCount()
    while True:
        if time_step(prev_t) < query_time_step:
            print("wait")
            time.sleep(query_time_step)
            continue
        else:
            count = 0
            start_t = cv2.getTickCount()
            batch_items = src_collection.get_batch_items(src_conditions, 'ct', batch_size=parallel_thread_num * 100)
            print("************Get %d items************" % len(batch_items))
            prev_t = cv2.getTickCount()
            if len(batch_items) == 0:
                continue
            for item in batch_items:
                task_queue.put(item)
                count += 1

            threads = []
            for thread_id in range(parallel_thread_num):
                threads.append(Thread(target=King_on_the_way, args=(thread_id,)))
                threads[-1].start()

            for thread in threads:
                thread.join()
            end_t = cv2.getTickCount()
            print("===============One Batch time: %.1f s with %d samples\n" % (
                (end_t - start_t) * 1.0 / cv2.getTickFrequency(), count))
