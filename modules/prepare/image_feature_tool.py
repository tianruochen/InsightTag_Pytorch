#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :image_feature_tool.py
# @Time     :2022/6/21 上午11:40
# @Author   :Chang Qing


import os
import cv2
import timm
import torch
import traceback

from PIL import Image
from collections import OrderedDict
from modules.model.efficientnet import Efficietnet_b5
from modules.dataset.argument import DataArgument
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

cur_script_dir = os.path.dirname(__file__)
IMAGETAGV6_EFFB5_BEST_CKPT = os.path.join(cur_script_dir, "image_extractor/weights/eff/effb5_cls103_best_online.pth")
VIT_LARGE_PATCH32_384_PATH = os.path.join(cur_script_dir, "video_extractor/weights/vit/jx_vit_large_p32_384-9b920ba8.pth")
# IMAGETAGV6_EFFB5_BEST_CKPT = '/home/work/changqing/Insight_Multimodal_Pytorch/modules/prepare/image_extractor/weights/eff/effb5_cls103_best_online.pth'
# VIT_LARGE_PATCH32_384_PATH = '/home/work/changqing/Insight_Multimodal_Pytorch/modules/prepare/video_extractor/weights/jx_vit_large_p32_384-9b920ba8.pth'


class ImageFeatureExtractor:
    def __init__(self, base_model_name="vit"):
        self.base_model_name = base_model_name

        self.n_gpu_available = torch.cuda.device_count()
        self.device = torch.device("cuda" if self.n_gpu_available > 0 else "cpu")
        self.model = self._build_model()
        self.dim = 2048 if self.base_model_name == "eff" else 1024
        self.tfms = self._build_transform()

    def _build_transform(self):
        if self.base_model_name == "eff":
            transform = DataArgument(image_resize=380)
        elif self.base_model_name == "vit":
            config = resolve_data_config({}, model=self.model)
            transform = create_transform(**config)
        else:
            raise NotImplementedError
        return transform

    def _build_model(self):
        if self.base_model_name == "eff":
            model = Efficietnet_b5(num_classes=103)
            model = model.to(self.device)
            checkpoint_path = IMAGETAGV6_EFFB5_BEST_CKPT
            checkpoint = torch.load(checkpoint_path)
            if self.n_gpu_available > 1:
                self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            else:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    if not k.startswith("module."):
                        break
                    name = k[7:]
                    new_state_dict[name] = v
                state_dict = new_state_dict if new_state_dict else checkpoint['state_dict']
                self.model.load_state_dict(state_dict, strict=True)
            model.eval()
        elif self.base_model_name == "vit":
            model = timm.create_model('vit_large_patch32_384', pretrained=False)
            model = model.cuda()
            checkpoint_path = VIT_LARGE_PATCH32_384_PATH
            model_data = torch.load(checkpoint_path)
            model.load_state_dict(model_data)
            model.eval()
        else:
            raise NotImplementedError
        return model

    def extract_features(self, image_paths, remove_image=False):

        image_features = torch.cuda.FloatTensor(0, self.dim).fill_(0)

        with torch.no_grad():
            for img_path in image_paths:
                try:
                    image = Image.open(img_path)
                    image = image.convert("RGB")

                    image_tensor = self.tfms(image).unsqueeze(0)
                    image_tensor = image_tensor.to(self.device)
                    if self.base_model_name == "eff":
                        logits, feature = self.model(image_tensor, return_feature=True)
                    else:
                        feature = self.model.forward_features(image_tensor)
                    image_features = torch.cat((image_features, feature), dim=0)
                    # feature = feature.cpu()
                except:
                    print(img_path)
                    traceback.print_exc()
                    pass
        image_features = image_features.cpu().numpy()

        if remove_image:
            for image_path in image_paths:
                if os.path.exists(image_path):
                    os.remove(image_path)
        if image_features.shape[0] == 0:
            return None
        return image_features
