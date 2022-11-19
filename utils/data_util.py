#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :data_util.py
# @Time     :2022/8/17 下午4:38
# @Author   :Chang Qing
 

import numpy as np


class IncreaseMeanStd:
    '''
    增量计算海量数据平均值和标准差,方差
    1.数据
        obj.mean为平均值
        obj.std为标准差
        obj.n为数据个数
        对象初始化时需要指定历史平均值,历史标准差和历史数据个数(初始数据集为空则可不填写)
    2.方法
    obj.incre_in_list()方法传入一个待计算的数据list,进行增量计算,获得新的mean,std和n(海量数据请循环使用该方法)
    obj.incre_in_value()方法传入一个待计算的新数据,进行增量计算,获得新的mean,std和n(海量数据请将每个新参数循环带入该方法)
    '''

    def __init__(self, history_mean=0, history_std=0, history_nums=0):
        self.mean = history_mean
        self.std = history_std
        self.nums = history_nums

    def incre_in_list(self, new_list):
        mean_new = np.mean(new_list)
        incre_mean = (self.nums * self.mean + len(new_list) * mean_new) / \
                    (self.nums + len(new_list))
        std_new = np.std(new_list, ddof=1)
        incre_std = np.sqrt((self.nums * (self.std ** 2 + (incre_mean - self.mean) ** 2) + len(new_list)
                                * (std_new ** 2 + (incre_mean - mean_new) ** 2)) / (self.nums + len(new_list)))
        self.mean = incre_mean
        self.std = incre_std
        self.nums += len(new_list)

    def incre_in_value(self, value):
        incre_mean = (self.nums * self.mean + value) / (self.nums + 1)
        incre_std = np.sqrt((self.nums * (self.std ** 2 + (incre_mean - self.mean)
                                          ** 2) + (incre_mean - value) ** 2) / (self.nums + 1))
        self.mean = incre_mean
        self.std = incre_std
        self.nums += 1

    def incre_in_array(self, new_array):
        mean_new = np.mean(new_array)
        data_num = new_array.size
        incre_mean = (self.nums * self.mean + data_num * mean_new) / \
                     (self.nums + data_num)
        std_new = np.std(new_array, ddof=1)
        incre_std = np.sqrt((self.nums * (self.std ** 2 + (incre_mean - self.mean) ** 2) + data_num
                             * (std_new ** 2 + (incre_mean - mean_new) ** 2)) / (self.nums + data_num))
        self.mean = incre_mean
        self.std = incre_std
        self.nums += data_num
    
    def report(self):
        return self.mean, self.std, self.nums


if __name__ == "__main__":
    c = IncreaseMeanStd()
    # c.incre_in_value(0.05)
    # print(c.mean)
    # print(c.std)
    # print(c.nums)
    # c.incre_in_value(0.02)
    # c.incre_in_list([0.5, 0.2, 0.3])
    # print(c.mean)
    # print(c.std)
    # print(c.nums)

    c.incre_in_array(np.zeros((3,2)))
    c.incre_in_array(np.ones((3,2)))

    print(c.mean)
    print(c.std)
    print(c.nums)


