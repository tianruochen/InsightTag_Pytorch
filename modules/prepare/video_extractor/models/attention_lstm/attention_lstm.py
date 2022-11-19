#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.fluid as fluid
from paddle.fluid import ParamAttr

from ..model import ModelBase
from .lstm_attention import LSTMAttentionModel

import logging

logger = logging.getLogger(__name__)

__all__ = ["AttentionLSTM"]


class AttentionLSTM(ModelBase):
    def __init__(self, name, cfg, mode='train', is_videotag=False):
        super(AttentionLSTM, self).__init__(name, cfg, mode)
        self.is_videotag = is_videotag
        self.get_config()

    def get_config(self):
        # get model configs
        self.feature_names = self.cfg.MODEL.feature_names    # ["rgb"]
        self.feature_dims = self.cfg.MODEL.feature_dims      # [2048]
        self.num_classes = self.cfg.MODEL.num_classes        # 3396
        self.embedding_size = self.cfg.MODEL.embedding_size  # 1024
        self.lstm_size = self.cfg.MODEL.lstm_size            # 512
        self.drop_rate = self.cfg.MODEL.drop_rate            # 0.5

        # get mode configs
        self.batch_size = self.get_config_from_sec(self.mode, 'batch_size', 1)   # 128
        self.num_gpus = self.get_config_from_sec(self.mode, 'num_gpus', 1)       # 1

        if self.mode == 'train':
            self.learning_rate = self.get_config_from_sec('train',
                                                          'learning_rate', 1e-3)  # 0.000125
            self.weight_decay = self.get_config_from_sec('train',
                                                         'weight_decay', 8e-4)    # 0.0008
            self.num_samples = self.get_config_from_sec('train', 'num_samples',
                                                        5000000)                  # x
            self.decay_epochs = self.get_config_from_sec('train',
                                                         'decay_epochs', [5])     # [5]
            self.decay_gamma = self.get_config_from_sec('train', 'decay_gamma',
                                                        0.1)                      # 0.1

    def build_input(self, use_dataloader):
        # print(self.mode)
        self.feature_input = []   # [[None, 2048]]
        for name, dim in zip(self.feature_names, self.feature_dims):
            self.feature_input.append(
                fluid.data(
                    shape=[None, dim], lod_level=1, dtype='float32', name=name))
        if self.mode != 'infer':
            # [None, 3396]
            # print(self.num_classes)
            self.label_input = fluid.data(
                shape=[None, self.num_classes], dtype='float32', name='label')
        else:
            self.label_input = None
        if use_dataloader:
            assert self.mode != 'infer', \
                'dataloader is not recommendated when infer, please set use_dataloader to be false.'
            self.dataloader = fluid.io.DataLoader.from_generator(
                feed_list=self.feature_input + [self.label_input],
                capacity=8,
                iterable=True)

    def build_model(self):
        att_outs = []
        for i, (input_dim, feature
                ) in enumerate(zip(self.feature_dims, self.feature_input)):
            # input_dim:[2048], feature:[[None,2048]]
            att = LSTMAttentionModel(input_dim, self.embedding_size,
                                     self.lstm_size, self.drop_rate)
            att_out = att.forward(feature, is_training=(self.mode == 'train'))
            att_outs.append(att_out)
        if len(att_outs) > 1:
            out = fluid.layers.concat(att_outs, axis=1)
        else:
            out = att_outs[0]  # video only, without audio in videoTag

        fc1 = fluid.layers.fc(
            input=out,
            size=8192,
            act='relu',
            bias_attr=ParamAttr(
                regularizer=fluid.regularizer.L2Decay(0.0),
                initializer=fluid.initializer.NormalInitializer(scale=0.0)),
            name='fc1')
        fc2 = fluid.layers.fc(
            input=fc1,
            size=4096,
            act='tanh',
            bias_attr=ParamAttr(
                regularizer=fluid.regularizer.L2Decay(0.0),
                initializer=fluid.initializer.NormalInitializer(scale=0.0)),
            name='fc2')

        ###### ours #######3
        self.embed8192 = fc1
        self.embed4096 = fc2
        print(self.num_classes)
        self.logit = fluid.layers.fc(input=fc2, size=self.num_classes, act=None, \
                                     bias_attr=ParamAttr(regularizer=fluid.regularizer.L2Decay(0.0),
                                                         initializer=fluid.initializer.NormalInitializer(scale=0.0)),
                                     name='output')

        # 多标签任务
        self.output = fluid.layers.sigmoid(self.logit)
        # 多分类任务
        # self.output = fluid.layers.softmax(self.logit)

    def optimizer(self):
        assert self.mode == 'train', "optimizer only can be get in train mode"
        values = [
            self.learning_rate * (self.decay_gamma ** i)
            for i in range(len(self.decay_epochs) + 1)
        ]
        iter_per_epoch = self.num_samples / self.batch_size
        boundaries = [e * iter_per_epoch for e in self.decay_epochs]
        return fluid.optimizer.RMSProp(
            learning_rate=fluid.layers.piecewise_decay(
                values=values, boundaries=boundaries),
            centered=True,
            regularization=fluid.regularizer.L2Decay(self.weight_decay))

    def loss(self):
        assert self.mode != 'infer', "invalid loss calculationg in infer mode"
        # 多标签任务
        cost = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=self.logit, label=self.label_input)
        cost = fluid.layers.reduce_sum(cost, dim=-1)
        # print("多标签")
        # print(self.label_input)
        # cost = fluid.layers.softmax_with_cross_entropy(logits=self.logit, label=self.label_input, soft_label=True)
        # cost = fluid.layers.reduce_sum(cost, dim=-1)
        sum_cost = fluid.layers.reduce_sum(cost)
        self.loss_ = fluid.layers.scale(
            sum_cost, scale=self.num_gpus, bias_after_scale=False)
        return self.loss_

    def outputs(self):
        return [self.output, self.logit]

    def feeds(self):
        return self.feature_input if self.mode == 'infer' else self.feature_input + [
            self.label_input
        ]

    def fetches(self):
        if self.mode == 'train' or self.mode == 'valid':
            losses = self.loss()
            fetch_list = [losses, self.output, self.label_input]
        elif self.mode == 'test':
            losses = self.loss()
            fetch_list = [losses, self.output, self.label_input]
        elif self.mode == 'infer':
            # fetch_list = [self.output]
            fetch_list = [self.output, self.embed4096, self.embed8192]
        else:
            raise NotImplementedError('mode {} not implemented'.format(
                self.mode))

        return fetch_list

    def weights_info(self):
        return None, None

    def load_pretrain_params(self, exe, pretrain, prog):
        logger.info("Load pretrain weights from {}, exclude fc layer.".format(
            pretrain))

        state_dict = fluid.load_program_state(pretrain)
        dict_keys = list(state_dict.keys())
        for name in dict_keys:
            if "fc_0" in name or "w_0" in name or "b_0" in name:
                del state_dict[name]
                logger.info(
                    'Delete {} from pretrained parameters. Do not load it'.
                        format(name))
        fluid.set_program_state(prog, state_dict)

