#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :argument.py
# @Time     :2021/4/1 下午9:14
# @Author   :Chang Qing

import torch
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class DataArgument:
    def __init__(self, image_resize=380, random_hflip=False, random_vflip=False, random_crop=False,
                 random_rotate=False, gaussianblur=False, random_erase=False, random_brightness=False,
                 random_gamma=False, random_saturation=False, random_adjust_hue=False):

        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        self.random_crop = random_crop
        self.gaussianblur = gaussianblur
        self.random_erase = random_erase
        self.image_resize = image_resize
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip
        self.random_gamma = random_gamma
        self.random_rotate = random_rotate
        self.random_brightness = random_brightness
        self.random_saturation = random_saturation
        self.random_adjust_hue = random_adjust_hue

    def __call__(self, image):
        # image = F.to_pil_image(image)
        # print(type(image))
        if self.random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.image_resize).get_params(img=image, scale=[0.5, 1.0],
                                                                                         ratio=[0.9, 1.1])
            image = TF.resized_crop(image, i, j, h, w, size=[self.image_resize, self.image_resize])
        else:
            image = TF.resize(image, [self.image_resize, self.image_resize], interpolation=3)

        if self.random_hflip and random.random() > 0.5:
            image = TF.hflip(image)
        if self.random_vflip and random.random() > 0.5:
            image = TF.vflip(image)
        if self.random_rotate and random.random() > 0.5:
            angle = random.randint(1, 4) * 90
            image = TF.rotate(image, angle)

        if self.random_brightness and random.random() > 0.5:
            # multiply a random number within a - b
            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.5, 1.5))

        if self.random_gamma and random.random() > 0.5:
            # img**gamma
            image = TF.adjust_gamma(image, gamma=random.uniform(0.5, 1.5))

        if self.random_saturation and random.random() > 0.5:
            # saturation_factor, 0: grayscale image, 1: unchanged, 2: increae saturation by 2
            image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.5, 1.5))

        if self.random_adjust_hue and random.random() > 0.5:
            # saturation_factor, 0: grayscale image, 1: unchanged, 2: increae saturation by 2
            hue_factor = random.random()-0.5
            image = TF.adjust_hue(image, hue_factor=hue_factor)

        if self.gaussianblur and random.random() > 0.5:
            image = TF.gaussian_blur(image, kernel_size=[3, 3])
        if self.random_erase and random.random() > 0.5:
            h = w = self.image_resize // 10
            i = j = random.randint(1, self.image_resize - 1)
            v = torch.tensor(self.img_mean) * 255
            image = TF.erase(image, i, j, h, w, v)
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.img_mean, self.img_std)

        return image


def build_transform(args):
    return DataArgument(**args)
