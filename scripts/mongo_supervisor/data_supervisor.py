#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import numpy as np
from DBVisitor import query_items
collection_config = {
    "input": {
        "host": "172.16.255.183",
        "port": 27047,
        "db": "multimodal_tag",
        "table": "multimodal_tag_input"
    },
    "output": {
        "host": "mongodb://root:b7sameEdxrfDiu3a@s-uf61da9362294a34.mongodb.rds.aliyuncs.com:3717/monet?authSource=admin",
        "port": 27047,
        "db": "monet",
        "table": "post_labels_v3"
    }
}


if __name__ == '__main__':
    input_param = collection_config['input']
    output_param = collection_config['output']

    input_param['conditions'] = [[]]
    input_samples = query_items(input_param)
    print("[Input] Total: ", len(input_samples))
    input_param['conditions'] = [['processed:=:False']]
    input_samples = query_items(input_param)
    print("[Input] Unprocessed: ", len(input_samples))

    output_param['conditions'] = [[]]
    output_samples = query_items(output_param)
    print("[Output] Total: ", len(output_samples))
    cnt = 0
    for sample in output_samples:
        if(sample['tag_details']):
            cnt += 1
    print("[Output] Predictions: ", cnt)
    for sample in output_samples[:10]:
        print(sample)
