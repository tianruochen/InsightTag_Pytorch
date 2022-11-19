#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :mongo_util.py
# @Time     :2022/10/20 下午7:50
# @Author   :Chang Qing

import sys
import bson
import json
import re

import datetime
import pymongo
from pymongo import MongoClient

mapping = {">": "$gt", "<": "$lt", "=": "$eq", "~": "$ne", "exist": "$exists"}


def parse_conditions(sets):
    if (len(sets) > 1):
        multi_cond = []
        for one_set in sets:
            multi_cond.append(parse_condition_set(one_set))
        return dict({"$or": multi_cond})
    else:
        return parse_condition_set(sets[0])


def parse_condition_set(one_set):
    cond_dict = dict()
    for condition in one_set:
        item, cond = parse_condition(condition)
        if item in cond_dict.keys():
            cond_dict[item].update(cond)
        else:
            cond_dict[item] = cond
    return cond_dict


def parse_condition(condition):
    assert (len(condition.split(':')) == 3)
    item, relation, comp = condition.split(':')
    relation = mapping[relation]
    comp = mapping_comp(comp)
    return item, dict({relation: comp})


def mapping_comp(comp):
    if sys.version_info > (3, 0):
        assert (type(comp) in [str, bytes])
    else:
        assert (type(comp) in [str])

    if isdigit(comp):
        return int(comp)

    if comp == 'True':
        return True
    if comp == 'False':
        return False

    rex_day = re.compile("^-[0-9]\d*days$")
    rex_h = re.compile("^-[0-9]\d*h$")
    if not ((comp in ['today', 'yesterday']) or bool(rex_day.match(comp)) or bool(rex_h.match(comp))):
        return comp

    now = datetime.datetime.now()
    time_ = None
    if comp == "today":
        time_ = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif comp == "yesterday":
        time_ = now.replace(hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=1)
    elif bool(rex_day.match(comp)):
        num = int(comp.split("days")[0].split("-")[-1])
        time_ = now.replace(hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=num)
    elif bool(rex_h.match(comp)):
        num = int(comp.split("h")[0].split("-")[-1])
        time_ = now - datetime.timedelta(hours=num)

    return int(time_.strftime('%s'))


def isdigit(elem):
    if type(elem) is int or str(elem).isdigit():
        return True

    if len(str(elem)) > 1:
        if str(elem)[0] == '-' and str(elem)[1:].isdigit():
            return True
    return False


class MongoCollection:
    def __init__(self, db_params):
        self.db_params = db_params
        self.collection = self._get_collection()

    def _get_collection(self):
        if 'durl' in self.db_params:
            client = MongoClient(self.db_params['durl'])
        else:
            if self.db_params["port"]:
                client = MongoClient(self.db_params["host"], int(self.db_params["port"]))
            else:
                client = MongoClient(self.db_params["host"])
        collection = client[self.db_params["db"]][self.db_params["table"]]
        return collection

    def insert_items(self, items):
        """
        增操作
        :param items: 添加的信息,可以是单个元素dict,也可以是多个元素list(dict)
        """
        if type(items) is dict:
            items = [items]
        for item in items:
            self.collection.update_one({'_id': item['_id']}, {'$set': item}, upsert=True)

    def update_one(self, pid, save_info):
        self.collection.update_one({'pid': bson.int64.Int64(pid)}, {'$set': save_info}, upsert=True)

    def delete_item(self, conditions):
        """
        删操作, 删除满足条件的单个元素
        :param conditions: 条件
        """
        self.collection.delete_one(conditions)

    def update_item(self, condition, key_pair):
        """
        改操作, 更新满足条件的单个元素
        :param params:
        :param condition:
        :param key_pair:
        """
        self.collection.update(condition, {'$set': key_pair})

    def query_items(self, conditions=None, _targets=None):
        """
        查操作
        :param conditions: 条件
        :param _targets: 关注的字段名
        :return: 查询结果
        """
        if conditions is None:
            conditions = [[]]
        if type(_targets) is str:
            targets = [_targets]
        elif _targets is None:
            targets = None
        else:
            targets = _targets

        conditions = parse_conditions(conditions)

        if targets is None:
            return list(self.collection.find(conditions))

        items = dict()
        for target in targets:
            items[target] = []

        for sample in self.collection.find(conditions):
            for target in targets:
                if target in sample.keys():
                    items[target].append(sample[target])
                else:
                    items[target].append(None)

        # rearrange the items into columns
        s_items = [items[target] for target in targets]
        if type(_targets) is str:
            return s_items[0]

        return s_items

    def get_batch_items(self, conditions, sort_key, batch_size=5):
        """
        从collection中查询batch个数据
        :param conditions: 查询条件
        :param sort_key:  按sort_key排序
        :param batch_size:  查询数量
        :return: 查询结果(list)
        """
        conditions = parse_conditions(conditions)
        items = []
        for _ in range(batch_size):
            item = self.collection.find_one_and_update(conditions, {'$set': {'processed': True}},
                                                       sort=[(sort_key, pymongo.DESCENDING)])
            if item:
                items.append(item)
        return items


if __name__ == "__main__":
    from utils.config_util import parse_config
    config_path = "/home/work/changqing/Insight_Multimodal_Pytorch/configs/multimodal_online_infer_pipe.yaml"
    config = parse_config(config_path)
    # db_src_params = config.database.db_src_params
    # src_collection_manager = MongoCollection(db_params=db_src_params)
    # samples = src_collection_manager.query_items()
    # print(len(samples))
    db_dst_params = config.database.db_dst_params
    print(db_dst_params)
    dst_collection_manager = MongoCollection(db_params=db_dst_params)
    samples = dst_collection_manager.query_items()
    print(len(samples))