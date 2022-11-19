# -*- coding:utf-8 -*-
import os
import json
import bson
import time
import logging
import datetime
import traceback
import cv2

from modules.solver.inferer import Inferer
from utils.collection_util import MongoCollection
from modules.solver.preprocessor import ModalPreprocessor


# FIXME logger 多线程问题, 日志只能输出到一个文件中

class King:
    def __init__(self, config, king_id=0):
        # 基础配置
        self.config = config
        self.king_id = king_id
        self.database = config.database
        self.log_dir = config.runner.log_dir
        self.filter_threshold = config.runner.filter_threshold
        self.second_label2idx = json.load(open(config.runner.second_label2idx, "r"))
        self.runtime_log_path = os.path.join(self.log_dir, f'multimodal_runtime_{king_id}.log')
        self.logger = self._setup_logger()
        # 数据库collection相关
        self.db_src_params = self.database.db_src_params
        self.db_dst_params = self.database.db_dst_params
        self.src_collection = MongoCollection(self.db_src_params)
        self.dst_collection = MongoCollection(self.db_dst_params)

        # 模型相关
        self.modal_preprocessor = ModalPreprocessor(self.config.runner.data_temp_dir)
        self.modal_predictor = Inferer(config)

    def _setup_logger(self):
        os.makedirs(self.log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            # format="[%(asctime)12s] [%(levelname)7s] (%(filename)15s:%(lineno)3s): %(message)s",
            format="[%(asctime)12s] [%(levelname)s] : %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.runtime_log_path)
            ]
        )
        return logging.getLogger(str(self.king_id))

    def kill(self, sample):
        # print(f"king id is: {self.king_id}")
        # print(self.logger)
        process_tik = sample['ct']
        model_forward_tik = time.time()
        pid = sample['_id']
        self.logger.info("=" * 50)
        self.logger.info(f">>> processing {sample['_id']}...")
        error_mess = "SUCCESSFUL"  # error_code=1

        # model forward and generate output
        tag_details, error_code = self.process(pid)

        # parse mode output
        if error_code == -1:
            error_mess = "ERROR! Mode forward error, please check!"
        process_tok = model_forward_tok = time.time()
        process_cost = str(round(process_tok - process_tik, 2)) + "s"
        model_forward_cost = str(round(model_forward_tok - model_forward_tik, 2)) + "s"
        # build results information
        upload_info = {
            "ct": bson.int64.Int64(time.time()),
            "date": str(datetime.datetime.now().strftime('%Y%m%d')),
            "tag_details": tag_details
        }
        # upload_info = {'_id': pid,
        #                'results': results,
        #                'process_tik': process_tik,
        #                "process_tok": process_tok,
        #                "process_cost": process_cost,
        #                "model_forward_tik": model_forward_tik,
        #                "model_forward_cost": model_forward_cost,
        #                "error_code": error_code,
        #                "error_mess": error_mess}

        # post-processing upload final results and delete temp results
        # upload final resutls
        # self.dst_collection.insert_items(upload_info)
        # delete the task in queue
        self.dst_collection.update_one(pid=sample['_id'], save_info=upload_info)

        del_conditions = {'_id': sample['_id']}
        self.src_collection.delete_item(del_conditions)

        self.logger.info(f">>> Done with {pid}, status: {error_code}-{error_mess}, "
                         f"process_cost: {process_cost}, model forward cost: {model_forward_cost}")
        self.logger.info("=" * 50)
        return

    def filter_results(self, results):
        new_results = {}
        labels_idx = []
        for label, score in results.items():
            if score > self.filter_threshold:
                new_results[label] = round(score, 4)
                second_label = label.split("_")[-1]
                labels_idx.append(self.second_label2idx[second_label])
        return new_results, labels_idx

    def process(self, pid):
        try:
            pre_tik = cv2.getTickCount()
            # 1.获取模态特征
            modal_feature = self.modal_preprocessor.preprocess(pid)
            pre_tok = cv2.getTickCount()
            self.logger.info(f"    ... {pid} 预处理耗时: {(pre_tok - pre_tik) / cv2.getTickFrequency()}")

            # 2.过模型获取结果
            inf_tik = cv2.getTickCount()
            results = self.modal_predictor.inference_item(modal_feature)
            name_score_dict = results
            tag_details = []
            if self.filter_threshold:
                name_score_dict, tag_details = self.filter_results(results)
            inf_tok = cv2.getTickCount()
            self.logger.info(f"    ... {pid} 过模型耗时: {(inf_tok - inf_tik) / cv2.getTickFrequency()}")
            self.logger.info(f"    ... {pid} 结果为: {name_score_dict}")
            # 3. return output
            return tag_details, 1
        except:
            traceback.print_exc()
            return [], -1


if __name__ == '__main__':
    import os
    from utils.config_util import parse_config

    config_path = "configs/multimodal_online_infer_pipe.yaml"
    config = parse_config(config_path)

    king = King(config, king_id=3)
    tags, code = king.process('307624437')
    print('Code: ', code)
    print(tags)
