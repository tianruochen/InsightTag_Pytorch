#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :trainer.py
# @Time     :2022/6/21 下午5:40
# @Author   :Chang Qing

import os
import json
import math
import torch
import random
import logging
import datetime
import numpy as np

from time import time
from collections import deque
from torch.utils.data import DataLoader

from collections import OrderedDict
from utils.comm_util import ResultsLog
from modules.solver.base import Base
from modules.loss import build_loss
from modules.metric import MultiLabel_Metric
from modules.dataset import MultiDataset, MultiModalDataset
from modules.model.pytorch_pretrained.optimization import BertAdam


class Trainer(Base):

    def __init__(self, config):
        self.config = config
        super().__init__(config.basic, config.arch)

        self._setup_training_env()

        # 冻结bert的浅层权重
        self._freeze_model_asr()
        # build dataset and dataloader
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.train_data_path = config.data.train_data_path
        # self.train_data = MultiDataset(self.train_data_path, self.device)
        self.train_data = MultiModalDataset(self.train_data_path, self.device)

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size,
                                       shuffle=True)

        self.do_validation = False
        self.valid_data_path = config.data.valid_data_path
        if self.valid_data_path:
            self.do_validation = True
            # self.valid_data = MultiDataset(self.valid_data_path, self.device)
            self.valid_data = MultiModalDataset(self.valid_data_path, self.device)
            self.valid_loader = DataLoader(self.valid_data, batch_size=self.batch_size,
                                           shuffle=True)

        # build loss
        self.loss = build_loss(self.config.loss, self.num_classes, device=self.device)

        # build metrics,
        self.metrics = MultiLabel_Metric(self.num_classes)

        # build optimizer
        self.epochs = self.config.runner.epochs
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'bn']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        self.optimizer = BertAdam(optimizer_grouped_parameters,
                                  lr=float(self.config.runner.optimizer.args.lr),
                                  warmup=self.config.runner.optimizer.args.warmup,
                                  t_total=len(self.train_loader) * self.epochs)

        # configuration to guide training
        self.save_freq = config.runner.save_freq
        self.verbosity = config.runner.verbosity
        self.log_interval = config.runner.log_interval

        self.start_epoch = 1
        self.max_iter = len(self.train_loader) * self.epochs

        # configuration to monitor model performance and save best
        self.monitor = self.config.runner.monitor
        self.monitor_mode = self.config.runner.monitor_mode
        assert self.monitor_mode in ['min', 'max', 'off']
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf

        # setup visualization writer instance
        self.results_log = ResultsLog()

        # save deque
        self.save_max = self.config.runner.save_max
        self.save_deque = deque()

        # use_ema
        self.use_ema = self.config.arch.use_ema
        self.ema_decay = self.config.arch.ema_decay
        if self.use_ema and self.ema_decay:
            from modules.model.ema import ModelEMA
            self.logger.info("Use ema model ... ")
            self.ema_model = ModelEMA(self.model, self.ema_decay, self.device)

    def _freeze_model_asr(self):
        self.logger.info("freeze_model_asr... ")
        unfreeze_layers = ['layer.9', 'layer.10', 'layer.11', 'out.']
        if len(self.device_ids) > 1:
            model_asr = getattr(self.model.module, 'model_asr')
        else:
            model_asr = getattr(self.model, 'model_asr')
        for name, param in model_asr.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

    def _setup_training_env(self):
        # Setup directory for checkpoint saving
        start_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
        # workshop/task_arch/datetime/ckpt/
        self.checkpoint_dir = os.path.join(os.path.abspath(self.config.runner.save_dir),
                                           self.task_name + "_" + start_time,
                                           self.config.runner.ckpt_dir)
        # print(self.checkpoint_dir)
        # workshop/task_arch/datetime/log/
        self.models_log_dir = os.path.join(os.path.abspath(self.config.runner.save_dir),
                                           self.task_name + "_" + start_time,
                                           self.config.runner.log_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.models_log_dir, exist_ok=True)

        # save config
        self.runtime_log_path = os.path.join(self.models_log_dir, 'runtime.log')
        self.logger.addHandler(logging.FileHandler(self.runtime_log_path))
        self.config_save_path = os.path.join(self.models_log_dir, 'config.json')
        self.results_log_path = os.path.join(self.models_log_dir, 'results_log.json')
        with open(self.config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=False)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("[EPOCH %d]" % (epoch))
            start_time = time()
            result = self._train_epoch(epoch)
            finish_time = time()
            self.logger.info(
                "Finish at {}, Runtime: {:.3f} [s]".format(datetime.datetime.now(), finish_time - start_time))

            # print logged informations to the screen
            if self.results_log is not None:
                self.results_log.add_entry(result)
                if self.verbosity >= 1:
                    self.logger.info(f"=====================The results of epoch:  {epoch} ===================")
                    for key, value in sorted(list(result.items())):
                        self.logger.info('              {:25s}: {}'.format(str(key), value))
                    self.logger.info(f"=============================Report Done ================================")
            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.monitor_mode != 'off':
                try:
                    if (self.monitor_mode == 'min' and result[self.monitor] < self.monitor_best) or \
                            (self.monitor_mode == 'max' and result[self.monitor] > self.monitor_best):
                        self.logger.info("Monitor improved from %f to %f" % (self.monitor_best, result[self.monitor]))
                        self.monitor_best = result[self.monitor]
                        best = True
                except KeyError:
                    if epoch == 1:
                        msg = "Warning: Can\'t recognize metric named '{}' ".format(self.monitor) \
                              + "for performance monitoring. model_best checkpoint won\'t be updated."
                        self.logger.warning(msg)

            # Save checkpoint
            self._save_checkpoint(epoch, save_best=best)
        logging.info("******************Training Done..*********************")
        self._save_results_log()

    def _save_results_log(self):
        with open(self.results_log_path, 'w') as f:
            f.write(json.dumps(self.results_log.entries, indent=4, sort_keys=True))

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        # Construct savedict
        # arch = type(self.model).__name__  DataParallel
        state = {
            'arch': self.config.arch.name,
            'epoch': epoch,
            'results_log': str(self.results_log),
            'state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.ema.state_dict() if self.use_ema else None,
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            # 'config': self.config
        }

        monitor_best = round(self.monitor_best, 3)
        # Save checkpoint for each epoch
        if self.save_freq is not None:  # Use None mode to avoid over disk space with large models
            if epoch % self.save_freq == 0:
                filename = os.path.join(self.checkpoint_dir,
                                        f"{self.task_name}_{self.config.arch.name}_epoch{epoch}_{self.monitor}{monitor_best}.pth")
                torch.save(state, filename)
                self.logger.info("Saving checkpoint at {}".format(filename))

        # Save the best checkpoint
        if save_best:
            if len(self.save_deque) >= self.save_max:
                need_removed_checkpoint = self.save_deque.popleft()
                os.remove(need_removed_checkpoint)
            checkpoint_name = f"{self.task_name}_{self.config.arch.name}_epoch{epoch}_{self.monitor}{monitor_best}.pth"  # arch + "_epoch" + str(epoch) + "_" + self.monitor
            best_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            torch.save(state, best_path)

            # save best weights, if ues_ema, save ema weights
            if self.use_ema:
                model_to_save = self.ema_model.ema
                best_weights_path = os.path.join(self.checkpoint_dir, "best_weights_with_ema.pth")
            else:
                model_to_save = self.model
                best_weights_path = os.path.join(self.checkpoint_dir, "best_weights.pth")

            torch.save(model_to_save.state_dict(), best_weights_path)
            self.save_deque.append(best_path)
            self.logger.info("Saving current best at {}".format(best_path))
        else:
            self.logger.info("Monitor is not improved from %f" % (self.monitor_best))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """

        # valid_log = self._valid_epoch(epoch)
        # print(valid_log)
        self.logger.info(f"*****************************Training on epoch {epoch}...*****************************")
        self.model.train()
        self.metrics.reset()

        # Perform training
        n_iter = len(self.train_loader)
        batch_count = len(self.train_loader)
        # (img_tensors, label_tensors, img_paths)
        for batch_idx, (text_, video_, audio_, label_, inv_ids) in enumerate(self.train_loader):
            # 将全0的tensor过滤掉 reference： https://github.com/pytorch/pytorch/issues/1206
            video_input = video_[0]
            audio_input = audio_[0]
            # text_asr = text_[0]
            if torch.equal(video_input, torch.zeros(video_input.shape).cuda()) or \
                    torch.equal(audio_input, torch.zeros(audio_input.shape).cuda()):
                self.logger.info("全0数据的inv_ids: {}".format(str(inv_ids)))
                continue

            # print(data.shape, target.shape)
            curr_iter = batch_idx + (epoch - 1) * n_iter
            # data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            combine_out, (video_out, audio_out, text_out) = self.model(text_, video_, audio_)  # (bs,1,384,384)
            have_error_data = torch.isnan(combine_out)
            if True in have_error_data:
                print("*" * 100)
                print("Error")
                print(inv_ids)
                print(label_)
                print(combine_out)
                print(inv_ids)
                print("*" * 100)
                continue
            # adaptive loss 会返回一个tuple
            loss = self.loss(combine_out, label_)
            if isinstance(loss, tuple):
                loss = loss[0]
            loss.backward()
            self.optimizer.step()

            # update ema model
            if self.use_ema and self.ema_model:
                self.ema_model.update(self.model)
            self.model.zero_grad()

            batch_accuracy, batch_matched = self.metrics.update(combine_out, label_, loss.item())
            # total_loss += loss.item()
            # total_metrics += self._eval_metrics(output, target)

            if curr_iter % self.log_interval == 0:
                self.logger.info(
                    "Epoch:{:3d} training batch:{:5}/{:5} -- loss:{:.4f} lr:{:.5f} batch_top@1_acc:{:.4f} specific：[{:3}/{:3}]".format(
                        epoch, batch_idx, batch_count, loss.item(),
                        self.optimizer.state_dict()['param_groups'][0]['lr'], batch_accuracy, batch_matched,
                        label_.shape[0]))
                # v_bn0 = np.mean(self.model.state_dict()['video_head.bn0.running_var'].cpu().numpy())
                # a_bn0 = np.mean(self.model.state_dict()['audio_head.bn0.running_var'].cpu().numpy())
                # t_bn0 = np.mean(self.model.state_dict()['text_head.bn0.running_var'].cpu().numpy())
                #
                # # self.logger.info("v_b0: {}, a_b0: {}, t_b0: {}".format(v_bn0, a_bn0, t_bn0))
                # if np.isnan(v_bn0) or np.isnan(a_bn0) or np.isnan(t_bn0):
                #     print("video_input" + "#" * 50)
                #     print(video_input)
                #     print(audio_input)
                #     print("audio_input" + "#" * 50)
                # if (curr_iter + 1) % (n_iter // 5000) == 0:
                #     gap, avg_loss, avg_acc, avg_auc, acc_for_class, auc_for_class = self.metrics.report()
                #     print(gap, avg_loss, avg_acc, avg_auc, acc_for_class, auc_for_class)
                #     temp_ckpt = self.model.state_dict()
                #     torch.save(temp_ckpt, f"/home/work/changqing/Insight_Multimodal_Pytorch/temp_models/{curr_iter}.pth")

        # Record log
        gap, avg_loss, avg_acc, avg_auc, acc_for_class, auc_for_class = self.metrics.report(top_k=5)

        log = {
            'train_gap': gap,
            'train_loss': avg_loss,
            'train_acc': avg_acc,
            "train_auc_for_class": auc_for_class,
            "train_auc": avg_auc,
            "train_acc_for_class": acc_for_class
        }
        # log = {}

        # Perform validating
        if self.do_validation:
            self.logger.info(
                f"*****************************Validation on epoch {epoch}...*****************************")
        val_log = self._valid_epoch(epoch)
        log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'valid_metrics'.
        """
        if self.use_ema:
            test_model = self.ema_model.ema
        else:
            test_model = self.model
        test_model.eval()
        self.metrics.reset()
        # total_val_loss = 0
        # total_val_metrics = np.zeros(len(self.metrics))
        n_iter = len(self.valid_loader)

        with torch.no_grad():
            # Validate
            for batch_idx, (text_, video_, audio_, label_, inv_ids) in enumerate(self.valid_loader):
                # data, target = data.to(self.device), target.to(self.device)
                combine_out, (video_out, audio_out, text_out) = test_model(text_, video_, audio_)
                have_error_data = torch.isnan(combine_out)
                if True in have_error_data:
                    print("*" * 100)
                    print("Error")
                    print(inv_ids)
                    print(label_)
                    print(combine_out)
                    print(inv_ids)
                    print("*" * 100)
                    continue
                loss = self.loss(combine_out, label_)
                # print(output, loss)
                self.metrics.update(combine_out, label_, loss.item())

        # Record log
        # if the task_type == "multi_label", the ava_acc is top@1_acc
        gap, avg_loss, avg_acc, avg_auc, acc_for_class, auc_for_class = self.metrics.report(top_k=5)

        val_log = {
            'valid_gap': gap,
            'valid_loss': avg_loss,
            'valid_acc': avg_acc,
            "valid_auc": avg_auc,
            "valid_acc_for_class": acc_for_class,
            "valid_auc_for_class": auc_for_class,

        }
        return val_log
