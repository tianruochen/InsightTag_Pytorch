import torch
import numpy as np
from modules.model.bert import Bert
from modules.model.NextVLAD import NeXtVLAD
from modules.model.STAM import STAM
from modules.model.transformer import Transformer
from modules.model.longformer import Longformer
from modules.model.cait import cait_XXS36_224
from modules.model.LMF_TWO import LMF
from modules.model.MULTModel import MULTModel
from modules.model.channel_attention import SELayer
from modules.model.classifier_head import Classifier_head
from modules.model.encoders import BiModalEncoder
from modules.model.eca_layer import eca_layer

import torch.nn as nn

import torch.nn.functional as F
import pandas as pd
df_all_label = pd.read_csv("/data1/zhanglei/multimodal_tag/label/alllabel2.csv")
label_size = len(df_all_label)


# 图片、视频、语音、文本多模态融合网络
class MultiFusionNet4D(nn.Module):
    def __init__(self, video_dim=1536, audio_dim=768, text_dim=768, image_dim=1024, num_classes=1000):
        super(MultiFusionNet4D, self).__init__()
        print('Inital Bi_trans_vit ocr model')
        self.video_dim = video_dim  # 1536
        self.audio_dim = audio_dim  # 768
        self.image_dim = image_dim  # 1024
        self.text_dim = text_dim  # 768
        self.num_classes = num_classes

        self.model_asr = Bert(768)

        self.video_fc = nn.Linear(1024, self.video_dim)
        self.audio_fc = nn.Linear(128, self.audio_dim)
        self.image_fc = nn.Linear(2048, self.image_dim)   #  图片的输入维度为（bs, 3, 2048）

        self.va_encoder = BiModalEncoder(audio_dim, video_dim, None, 0.1, 12, 3072, 3072, 2)
        self.vt_encoder = BiModalEncoder(text_dim, video_dim, None, 0.1, 12, 3072, 3072, 2)
        self.at_encoder = BiModalEncoder(audio_dim, text_dim, None, 0.1, 12, 3072, 3072, 2)

        # self.iv_encoder = BiModalEncoder(video_dim, image_dim, None, 0.1, 12, 3072, 3072, 2)
        self.ia_encoder = BiModalEncoder(audio_dim, image_dim, None, 0.1, 12, 3072, 3072, 2)
        self.it_encoder = BiModalEncoder(audio_dim, image_dim, None, 0.1, 12, 3072, 3072, 2)

        self.video_head = NeXtVLAD(dim=video_dim, num_clusters=128, lamb=2, groups=16, max_frames=300)
        self.audio_head = NeXtVLAD(dim=audio_dim, num_clusters=128, lamb=2, groups=16, max_frames=120)
        self.image_head = NeXtVLAD(dim=image_dim, num_clusters=128, lamb=2, groups=16, max_frames=3)
        self.text_head = NeXtVLAD(dim=text_dim, num_clusters=128, lamb=2, groups=16, max_frames=128)

        self.video_att = SELayer(video_dim * 16)
        self.audio_att = SELayer(audio_dim * 16)
        self.image_att = SELayer(image_dim * 16)
        self.text_att = SELayer(text_dim * 16)

        self.video_out_fc = nn.Linear(self.video_dim * 16, self.num_classes)
        self.audio_out_fc = nn.Linear(self.audio_dim * 16, self.num_classes)
        self.image_out_fc = nn.Linear(self.image_dim * 16, self.num_classes)
        self.text_out_fc = nn.Linear(self.text_dim * 16, self.num_classes)

    def forward(self, text_, video_, audio_, image_):
        outputs_video = video_[0]
        outputs_audio = audio_[0]
        outputs_image = image_[0]     # (bs, 3, 2048)
        text_asr = text_[0]

        pool_outputs_asr, sequence_output_asr = self.model_asr(text_asr)
        outputs_video = self.video_fc(outputs_video)
        outputs_image = self.image_fc(outputs_image)      # (bs, 3, 1024)
        outputs_audio = self.audio_fc(outputs_audio)

        va_masks = {'A_mask': audio_[1].unsqueeze(1), 'V_mask': video_[1].unsqueeze(1)}
        vt_masks = {'A_mask': text_asr[2].unsqueeze(1), 'V_mask': video_[1].unsqueeze(1)}
        at_masks = {'A_mask': audio_[1].unsqueeze(1), 'V_mask': text_asr[2].unsqueeze(1)}
        # add
        # 基于显存的顾虑，所以没有做视频与图片之间的融合
        ia_masks = {'A_mask': audio_[1].unsqueeze(1), 'V_mask': image_[1].unsqueeze(1)}
        it_masks = {'A_mask': text_asr[2].unsqueeze(1), 'V_mask': image_[1].unsqueeze(1)}

        out_va = self.va_encoder((outputs_audio, outputs_video), va_masks)
        out_vt = self.vt_encoder((sequence_output_asr, outputs_video), vt_masks)
        out_at = self.at_encoder((outputs_audio, sequence_output_asr), at_masks)
        out_ia = self.ia_encoder((outputs_audio, outputs_image), ia_masks)
        out_it = self.it_encoder((sequence_output_asr, outputs_image), it_masks)

        va_audio_out = out_va[0]
        va_video_out = out_va[1]

        vt_text_out = out_vt[0]
        vt_video_out = out_vt[1]

        at_audio_out = out_at[0]
        at_text_out = out_at[1]

        ia_audio_out = out_ia[0]
        ia_image_out = out_ia[1]

        it_text_out = out_it[0]
        it_image_out = out_it[1]

        video_out = va_video_out + vt_video_out
        audio_out = va_audio_out + at_audio_out + ia_audio_out
        text_out = vt_text_out + at_text_out + it_text_out
        image_out = ia_image_out + it_image_out


        video_out = self.video_head(video_out).unsqueeze(2)
        audio_out = self.audio_head(audio_out).unsqueeze(2)
        # print(text_out.shape)
        text_out = self.text_head(text_out).unsqueeze(2)
        image_out = self.image_head(image_out).unsqueeze(2)

        audio_out = self.audio_att(audio_out).squeeze(2)
        video_out = self.video_att(video_out).squeeze(2)
        text_out = self.text_att(text_out).squeeze(2)
        image_out = self.image_att(image_out).squeeze(2)

        video_out = self.video_out_fc(video_out)
        video_out = video_out.float()
        video_out = torch.sigmoid(video_out)

        text_out = self.text_out_fc(text_out)
        text_out = text_out.float()
        text_out = torch.sigmoid(text_out)

        audio_out = self.audio_out_fc(audio_out)
        audio_out = audio_out.float()
        audio_out = torch.sigmoid(audio_out)
        # print(audio_out.shape)

        image_out = self.audio_out_fc(image_out)
        image_out = image_out.float()
        image_out = torch.sigmoid(image_out)

        combine_out_ = (video_out + text_out + audio_out + image_out) / 4
        return combine_out_

