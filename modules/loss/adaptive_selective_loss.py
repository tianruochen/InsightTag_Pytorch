#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :adaptive_selective_loss.py
# @Time     :2022/1/4 上午11:29
# @Author   :Chang Qing

import torch
import numpy as np
import torch.nn as nn


class AdaptiveSelectiveLoss(nn.Module):
    def __init__(self, class_num, device, alpha_pos=2, alpha_neg=1, alpha_unann=1, gamma_pos=0, gamma_neg=1, gamma_unann=2,
                 neg_plus=0, warmup_epoch=5, pseudo_label=True, pos_threshold=0.98, neg_threshold=0.01,
                 target_weights=None, likelihood_topk=5, classes_prior=None, prior_threshold=0.05,
                 compute_prior=False, partial_loss_mode=None, reduction="mean"):
        super(AdaptiveSelectiveLoss, self).__init__()
        self.class_num = class_num
        self.neg_plus = neg_plus
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.alpha_unann = alpha_unann
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.gamma_unann = gamma_unann

        if pseudo_label:
            self.pseudo_label = pseudo_label
            self.warmup_epoch = warmup_epoch
            self.pos_threshold = pos_threshold
            self.neg_threshold = neg_threshold

        self.device = device
        self.target_weights = target_weights
        if classes_prior is not None:
            self.classes_prior = classes_prior
            self.classes_prior.to(self.device)

        self.prior_threshold = prior_threshold
        self.compute_prior = compute_prior
        if not classes_prior and self.compute_prior:
            self.prior_computer = PriorComputer(num_classes=self.class_num)

        # likelihood_topk 未标注数据中 最有可能是某一个具体类型的topk，
        # 这topk个数据应该忽略掉，不应该作为负样本参与损失计算
        self.likelihood_topk = likelihood_topk
        self.partial_loss_mode = partial_loss_mode
        self.reduction = reduction

    def _build_targets_weights(self, targets, neg_prob, classes_prior=None):
        targets_weights = None
        if self.partial_loss_mode == "negative":
            # set all unsure targets as negative
            targets_weights = 1.0
        elif self.partial_loss_mode in ["ignore", None]:
            # remove all unsure targets
            targets_weights = torch.ones(targets.shape).to(self.device)
            targets_weights[targets == -1] = 0
        elif self.partail_loss_mode == "ignore_and_normalize":
            # remove all unsure targets and normalize by Durand et al. https://arxiv.org/pdf/1902.09720.pdfs
            # normalize
            alpha_norm, beta_norm = 1, 1
            targets_weights = torch.ones(targets.shape).to(self.device)
            # (bs,)
            n_annotated = 1 + torch.sum(targets != -1, axis=-1)  # add 1 to avoid dividing by zero
            g_norm = alpha_norm * (1 / n_annotated) + beta_norm
            classes_num = targets_weights.shape[1]
            # g_norm: (bs,) --> (1, bs) --> (classes_num, bs) --> (bs, classes_num)
            targets_weights *= g_norm.repeat([classes_num, 1]).T
            # ignore
            targets_weights[targets == -1] = 0
        elif self.partial_loss_mode == "adaptive_selective":
            # adaptive selective
            targets_weights = torch.ones(targets.shape).to(self.device)
            likelihood_batch_top_k = self.likelihood_topk * targets_weights[0]

            if classes_prior is not None:
                classes_prior = torch.tensor(list(classes_prior)).cuda()
                # 如果一个类型占比过高，且没有标签，则应该忽略掉
                idx_ignore = torch.where(classes_prior > self.prior_threshold)[0]
                targets_weights[:, idx_ignore] = 0
                targets_weights += (targets != -1).float()
                targets_weights = targets_weights.bool()

            # 根据得分过滤无标注的数据。未标注数据中 最有可能是某一个具体类型的topk，
            # 这topk个数据应该忽略掉，不应该作为负样本参与损失计算
            with torch.no_grad():
                targets_flatten = targets.flatten()
                cond_flatten = torch.where(targets_flatten == -1)[0]
                neg_prob_flatten = neg_prob.flatten()
                targets_weights_flatten = targets_weights.flatten()
                ind_class_sort = torch.argsort(neg_prob_flatten(cond_flatten))
                targets_weights_flatten[cond_flatten[ind_class_sort[:likelihood_batch_top_k]]] = 0

        return targets_weights

    def forward(self, logits, targets, cur_epoch=0):
        """
        :param logits:
        :param targets:
        :param classes_prior: 由模型计算的每个类型在总数据量中的数量占比 tensor (classes_num,)
        :return:
        """
        # Activation
        # pos_prob = torch.sigmoid(logits)  # 正样本的概率
        pos_prob = logits
        # print(torch.argmax(pos_prob[0], dim=-1), torch.where(targets[0] == 1))
        neg_prob = 1. - pos_prob  # 负样本的概率

        # compute prior
        classes_prior = None
        if self.compute_prior:
            self.prior_computer.update(pos_prob)
            classes_prior = self.prior_computer.get_prior(return_tensor=True)

        # neg plus
        if self.neg_plus is not None and self.neg_plus > 0.:
            neg_prob.add_(self.neg_plus).clamp_(max=1)

        # use pseudo label
        if self.pseudo_label and cur_epoch > self.warmup_epoch:
            targets[pos_prob > self.pos_threshold] = 1
            targets[pos_prob < self.neg_threshold] = 0

        # Positive, Negative and Un-annotated indexes
        targets_pos = (targets == 1).float()
        targets_neg = (targets == 0).float()
        targets_unann = (targets == -1).float()

        # build target weights
        targets_weights = self._build_targets_weights(targets, neg_prob, classes_prior)


        # calculate loss  (将未标注的数据当坐负样本，后面在进行过滤)
        BCE_pos = self.alpha_pos * targets_pos * torch.log(torch.clamp(pos_prob, min=1e-8))
        BCE_neg = self.alpha_neg * targets_neg * torch.log(torch.clamp(neg_prob, min=1e-8))
        BCE_unann = self.alpha_unann * targets_unann * torch.log(torch.clamp(neg_prob, min=1e-8))

        BCE_loss = BCE_pos + BCE_neg + BCE_unann

        # Adding asymmetric gamma weights   正样本的损失 > 负样本的损失 > 无标注样本的损失
        with torch.no_grad():
            asymmetric_w = torch.pow(1 - pos_prob * targets_pos - neg_prob * (targets_neg + targets_unann),
                                     self.gamma_pos * targets_pos + self.gamma_neg * targets_neg +
                                     self.gamma_unann * targets_unann)
        BCE_loss *= asymmetric_w

        targets_weights = targets_weights.to(BCE_loss.device)

        # partial labels weights
        BCE_loss *= targets_weights

        # reduction
        if self.reduction == "none":
            BCE_loss = -BCE_loss.sum()
        elif self.reduction == "mean":
            BCE_loss = -BCE_loss.mean()
        elif self.reduction == "sum":
            BCE_loss = -BCE_loss.sum()

        if self.compute_prior:
            return BCE_loss, classes_prior
        return BCE_loss




class PriorComputer:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.avg_pred = torch.zeros(num_classes).to(device)
        self.sum_pred = torch.zeros(num_classes).to(device)
        self.cnt_samples = 0.

    def update(self, pos_prob):
        with torch.no_grad():
            # preds = torch.sigmoid(logits).detach()
            self.sum_pred += torch.sum(pos_prob, axis=0)
            self.cnt_samples += pos_prob.shape[0]
            self.avg_pred = self.sum_pred / self.cnt_samples

    def get_prior(self, return_tensor=True):
        return self.avg_pred if return_tensor else self.avg_pred.cpu().tolist()

    # def get_top_freq_classes(self):
    #     n_top = 10
    #     top_idx = torch.argsort(-self.avg_pred.cpu())[:n_top]
    #     top_classes = np.array(list(self.classes.values()))[top_idx]
    #     print('Prior (train), first {} classes: {}'.format(n_top, top_classes))

