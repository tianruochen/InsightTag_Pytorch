import logging
import sys, os
import json
import re

import datetime
import pymongo
from pymongo import MongoClient

mapping = {">":"$gt", "<":"$lt", "=":"$eq", "~":"$ne", "exist":"$exists"}

def isdigit(elem):
    if(type(elem) is int or str(elem).isdigit()):
        return True
    
    if(len(str(elem)) > 1):
        if(str(elem)[0] == '-' and str(elem)[1:].isdigit()):
            return True
    return False

def mapping_comp(comp):
    if(isdigit(comp)):
        return int(comp)

    if(comp == 'True'):
        return True
    if(comp == 'False'):
        return False

    rex_day = re.compile("^-[0-9]\d*days$")
    rex_h   = re.compile("^-[0-9]\d*h$")
    if not ((comp in ['today','yesterday']) or bool(rex_day.match(comp)) or bool(rex_h.match(comp))):
        return comp
    
    now = datetime.datetime.now()
    if comp == "today":
        time = now.replace(hour=0,minute=0,second=0,microsecond=0)
    elif comp == "yesterday":
        time = now.replace(hour=0,minute=0,second=0,microsecond=0)-datetime.timedelta(days=1)
    elif bool(rex_day.match(comp)):
        num = int(comp.split("days")[0].split("-")[-1])
        time = now.replace(hour=0,minute=0,second=0,microsecond=0)-datetime.timedelta(days=num)
    elif bool(rex_h.match(comp)):
        num = int(comp.split("h")[0].split("-")[-1])
        time = now-datetime.timedelta(hours=num)

    
    return int(time.strftime('%s'))

def parse_condition(condition):
    assert(len(condition.split(':')) ==3)
    item, relation, comp = condition.split(':')
    relation = mapping[relation]
    comp = mapping_comp(comp)
    return item, dict({relation:comp})

def parse_condition_set(one_set):
    cond_dict = dict()
    for condition in one_set:
        item, cond = parse_condition(condition)
        if item in cond_dict.keys():
            cond_dict[item].update(cond)
        else:
            cond_dict[item] = cond
    return cond_dict

# bob need 'and' condition, so change 'or' to 'and'
def parse_conditions(sets):
    if(len(sets) > 1):
        multi_cond = []
        for one_set in sets:
            multi_cond.append(parse_condition_set(one_set))
        return dict({"$or":multi_cond})
    else:
        return parse_condition_set(sets[0])

def get_batch_items(params, sort_key, batch_size=5):
    host, port, database, table, conditions = params['host'], params['port'], \
                                              params['db'], params['table'], params['conditions']

    conditions = parse_conditions(conditions)
    client = MongoClient(host, int(port))
    collection = client[database][table]

    items = []
    for _ in range(batch_size):
        item = collection.find_one_and_update( \
                     conditions, \
                     {'$set': {'processed':True}}, \
                     sort=[(sort_key, pymongo.DESCENDING)])
        if(item):
            items.append(item)
    return items


def query_items(params, _targets=None):
    if(type(_targets) is str):
        targets = [_targets]
    elif(_targets is None):
        targets = None
    else:
        targets = _targets
        

    host, port, database, table, conditions = params['host'], params['port'], \
                                              params['db'], params['table'], params['conditions']

    conditions = parse_conditions(conditions)

    client = MongoClient(host, int(port))
    collection = client[database][table]

    if targets is None:
        return list(collection.find(conditions))

    items = dict()
    for target in targets:
        items[target] = []

    for sample in collection.find(conditions):
        for target in targets:
            if target in sample.keys():
                items[target].append(sample[target])
            else:
                items[target].append(None)

    # rearrange the items into columns
    s_items = [items[target] for target in targets]
    if(type(_targets) is str):
        return s_items[0]

    return s_items


def query_100(params, _targets=None):
    if (type(_targets) is str):
        targets = [_targets]
    elif (_targets is None):
        targets = None
    else:
        targets = _targets

    host, port, database, table, conditions = params['host'], params['port'], \
                                              params['db'], params['table'], params['conditions']

    conditions = parse_conditions(conditions)

    client = MongoClient(host, int(port))
    collection = client[database][table]

    if targets is None:
        return list(collection.find(conditions).sort('ct', pymongo.DESCENDING).limit(100))

    items = dict()
    for target in targets:
        items[target] = []

    for sample in collection.find(conditions):
        for target in targets:
            if target in sample.keys():
                items[target].append(sample[target])
            else:
                items[target].append(None)

    # rearrange the items into columns
    s_items = [items[target] for target in targets]
    if (type(_targets) is str):
        return s_items[0]

    return s_items

def insert_item(params, info):
    host, port, database, table = params['host'], params['port'], \
                                     params['db'], params['table']

    client = MongoClient(host, int(port))
    tb = client[database][table]
    if(type(info) is dict):
        _id = tb.update_one({'_id':info['_id']}, {'$set':info}, upsert=True)
    elif(type(info) is list):
        for _ in info:
            _id = tb.update_one({'_id':_['_id']}, {'$set':_}, upsert=True)

    return _id

def update_item(params, condition, key_pair):
    host, port, database, table = params['host'], params['port'], \
                                     params['db'], params['table']

    client = MongoClient(host, int(port))
    tb = client[database][table]
    tb.update( condition, {'$set':key_pair} )

    return 0

def delete_item(params, key_pair):
    host, port, database, table = params['host'], params['port'], \
                                     params['db'], params['table']

    client = MongoClient(host, int(port))
    tb = client[database][table]
    tb.delete_one(key_pair)
    return 0

def drop_table(params):
    host, port, database, table = params['host'], params['port'], \
                                     params['db'], params['table']

    client = MongoClient(host, int(port))
    db = client[database]
    db.drop_collection(table)
    return 0

