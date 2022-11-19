#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :server_util.py
# @Time     :2022/10/12 下午7:21
# @Author   :Chang Qing
 
import os
import uuid
import time
import hmac
import hashlib
import base64
import urllib

import datetime
from flask import jsonify
from pymongo import MongoClient


def error_resp(error_code, error_message):
    resp = jsonify(error_code=error_code, error_message=error_message)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


def build_task_input(task_key, redis_intance):
    task_input = task_key
    if "http" not in task_input:
        task_input = redis_intance.get(task_key)
    return task_input


def log_info(text):
    with open("skymagic_service_log.txt", "a") as f:
        f.write('%s' % datetime.datetime.now())
        f.write('    ')
        f.write(text)
        f.write('\n')
    return


def get_connection(db_params):
    host, port = db_params["host"], db_params["port"]
    db_name, tb_name = db_params["database"], db_params["table"]
    client = MongoClient(host, int(port))
    database = client[db_name]
    table = client[tb_name]
    return table


def write2db(db_params, info):
    collection = get_connection(db_params)
    if type(info) is dict:
        _id = collection.update_one({'_id': info['_id']}, {'$set': info}, upsert=True)
    elif type(info) is list:
        for _ in info:
            _id = collection.update_one({'_id': _['_id']}, {'$set': _}, upsert=True)
    return _id


def gen_signature(secret):
    timestamp = round(time.time() * 1000)
    secret_enc = secret.encode('utf-8')
    string_to_sign = '{}\n{}'.format(timestamp, secret)
    string_to_sign_enc = string_to_sign.encode('utf-8')
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    return timestamp, sign


def check_security(timestamp, sign, secrets):
    cur_timestamp = time.time() - timestamp
    # time out
    if cur_timestamp - timestamp > 60:
        return False, None
    for secret in secrets:
        # generate candidate sign
        secret_encode = secret.encode("utf-8")
        str_sign = "{}\n{}".format(timestamp, secret)
        str_sign_encode = str_sign.encode("utf-8")
        hmac_code = hmac.new(secret_encode, str_sign_encode, digestmod=hashlib.sha256).digest()
        candidate_sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        # match
        if candidate_sign == sign:
            return True, secret
    return False, None


