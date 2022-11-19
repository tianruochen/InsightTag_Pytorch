import torch
import torch.nn as nn

from modules.model.NextVLAD import NeXtVLAD
from modules.model.bert import Bert
from modules.model.channel_attention import SELayer
from modules.model.encoders import BiModalEncoder

"""
将ocr结果 作为单独的一个对象进行处理
"""


class MultiFusionNet(nn.Module):
    def __init__(self, video_dim=1536, audio_dim=768, text_dim=768, num_classes=1000):
        super(MultiFusionNet, self).__init__()
        print('Inital Bi_trans_vit ocr model')
        self.video_dim = video_dim       # 1536
        self.audio_dim = audio_dim       # 768
        self.text_dim = text_dim         # 768
        self.num_classes = num_classes
        
        self.model_asr = Bert(768)
        
        self.video_fc = nn.Linear(1024, self.video_dim)
        self.audio_fc = nn.Linear(128, self.audio_dim)
        
        self.va_encoder = BiModalEncoder(audio_dim, video_dim, None, 0.1, 12, 3072, 3072, 2)
        self.vt_encoder = BiModalEncoder(text_dim, video_dim, None, 0.1, 12, 3072, 3072, 2)
        self.at_encoder = BiModalEncoder(audio_dim, text_dim, None, 0.1, 12, 3072, 3072, 2)
        
        self.video_head = NeXtVLAD(dim=video_dim, num_clusters=128,lamb=2, groups=16, max_frames=210)
        
        self.audio_head = NeXtVLAD(dim=audio_dim, num_clusters=128,lamb=2, groups=16, max_frames=120)
        
        self.text_head = NeXtVLAD(dim=text_dim, num_clusters=128,lamb=2, groups=16, max_frames=128)

        self.video_att = SELayer(video_dim*16)
        self.audio_att = SELayer(audio_dim*16)
        self.text_att = SELayer(text_dim*16)

        self.video_out_fc = nn.Linear(self.video_dim*16, self.num_classes)
        self.audio_out_fc = nn.Linear(self.audio_dim*16, self.num_classes)
        self.text_out_fc = nn.Linear(self.text_dim*16, self.num_classes)


    def forward(self, text_, video_, audio_):
        outputs_video = video_[0]
        outputs_audio = audio_[0]

        # 单独处理图片ocr信息
        text_asr = text_[0]
        text_ocr = text_[1]
        # print(text_asr)

        pool_outputs_asr, sequence_output_asr = self.model_asr(text_asr)
        pool_outputs_ocr, sequence_output_ocr = self.model_asr(text_ocr)
        outputs_video = self.video_fc(outputs_video)
        outputs_audio = self.audio_fc(outputs_audio)

        va_masks = {'A_mask':audio_[1].unsqueeze(1), 'V_mask':video_[1].unsqueeze(1)}
        vt_masks = {'A_mask':text_asr[2].unsqueeze(1), 'V_mask':video_[1].unsqueeze(1)}
        at_masks = {'A_mask':audio_[1].unsqueeze(1), 'V_mask':text_asr[2].unsqueeze(1)}
        # 添加ocr的处理
        vt_o_masks = {'A_mask': text_ocr[2].unsqueeze(1), 'V_mask': video_[1].unsqueeze(1)}
        at_o_masks = {'A_mask': audio_[1].unsqueeze(1), 'V_mask': text_ocr[2].unsqueeze(1)}

        out_va = self.va_encoder((outputs_audio, outputs_video), va_masks)
        out_vt = self.vt_encoder((sequence_output_asr, outputs_video), vt_masks)
        out_at = self.at_encoder((outputs_audio, sequence_output_asr), at_masks)

        out_vt_o = self.vt_encoder((sequence_output_ocr, outputs_video), vt_o_masks)
        out_at_o = self.at_encoder((outputs_audio, sequence_output_ocr), at_o_masks)

        va_audio_out = out_va[0]
        va_video_out = out_va[1]
        
        vt_text_out = out_vt[0]
        vt_video_out = out_vt[1]

        at_audio_out = out_at[0]
        at_text_out = out_at[1]

        vt_o_text_out = out_vt_o[0]
        vt_o_video_out = out_vt_o[1]

        at_o_audio_out = out_at_o[0]
        at_o_text_out = out_at_o[1]

        video_out = va_video_out + vt_video_out + vt_o_video_out
        audio_out = va_audio_out + at_audio_out + at_o_audio_out
        text_out = vt_text_out + at_text_out + vt_o_text_out + at_o_text_out

        video_out = self.video_head(video_out).unsqueeze(2)
        audio_out = self.audio_head(audio_out).unsqueeze(2)
        # print(text_out.shape)
        text_out = self.text_head(text_out).unsqueeze(2)

        audio_out = self.audio_att(audio_out).squeeze(2)
        video_out = self.video_att(video_out).squeeze(2)
        text_out = self.text_att(text_out).squeeze(2)

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

        combine_out_ = (video_out + text_out + audio_out) / 3
        return combine_out_, (video_out, audio_out, text_out)



