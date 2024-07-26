import this
import scipy.io
import scipy.ndimage
import os
import os.path as osp
import json
import math
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError
from detectron2.structures import BoxMode
from PIL import Image
import cv2
import numpy as np
import random
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from os.path import splitext


from .datasets.cpn import CPNTargetGenerator
from .datasets import cpn


class CellDatasetMapper_SS(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        print("for cell")
        print(cfg.all_iter, cfg.e_sche, cfg.samples, cfg.img_file, cfg.data_file, cfg.transforms, cfg.item)
        # Rebuild augmentations
        # print(cfg)
        # print(cfg.items)
        self.items = None if cfg.item <= 0 else cfg.item
        # print(self.items)
        # assert 1 == 0
        self.split = 'train' if is_train else 'test'
        # self.split = 'train'
        # self.split = 'val'
        # self.split = 'train'
        self.training = is_train

        self.is_cos = cfg.is_cos
        self.use_freq_guild = cfg.use_freq_guild

        print("训练模式") if is_train else print("测试模式")

        color_jitter = transforms.ColorJitter(0.3, 0.3, 0.2)
        self.transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.5),
                transforms.RandomGrayscale(p=0.2),
                GussianBlur(kernel_size=3),
            ]
        )

        self.height = 224
        self.width = 224

        self.img_root_ss = cfg.img_file_ss
        self.load_all_img_list()

    def load_all_img_list(self):
        self.img_lists = os.listdir(self.img_root_ss)
        self.items = len(self.img_lists)

    def load_patches(self, image_name):

        ori_img = cv2.imread(os.path.join(self.img_root, image_name), 3)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        gray_im = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
        dft = cv2.dft(np.float32(gray_im), flags=cv2.DFT_COMPLEX_OUTPUT)
        dftshift = np.fft.fftshift(dft)
        # remove low fre
        mask = np.ones(dftshift.shape, np.uint8)
        crow, ccol = int(dftshift.shape[0] / 2), int(dftshift.shape[1] / 2)
        # radius = int(0.0005*dftshift.shape[0]/2)
        radius = 1

        cv2.circle(mask, (crow, ccol), radius, 0, -1)
        low_shift = dftshift * mask

        low_ishift = np.fft.ifftshift(low_shift)
        lowiimg = cv2.idft(low_ishift)
        # res_low = 1000*np.log(cv2.magnitude(lowiimg[:, :, 0], lowiimg[:, :, 1]))
        res_low = 10000 * (cv2.magnitude(lowiimg[:, :, 0], lowiimg[:, :, 1]))
        # res_low = (5*(cv2.magnitude(lowiimg[:, :, 0], lowiimg[:, :, 1])))**2
        res_low1 = res_low / float(np.max(res_low))
        res_low1[res_low1 <= 0.1] = 0.0

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bin_clo = cv2.erode(res_low1, kernel1, iterations=3)
        bin_clo[bin_clo > 0] = 1
        num_labels, labels, stats, centriods = cv2.connectedComponentsWithStats(np.uint8(bin_clo), connectivity=8)
        positive_patches = []
        positive_gmask = []

        negative_patches = []
        negative_gmask = []

        tmask = np.zeros_like(bin_clo)
        for idx, e in enumerate(stats):
            assert len(e) == 5
            tlx, tly, w, h, area = e
            if area <= 40:
                continue
            if w * h >= 8000 or w * h <= 40:
                continue

            sgray_img = gray_im[tly: tly + h, tlx: tlx + w]
            # print(np.mean(sgray_img))
            if np.mean(sgray_img) >= 135: # background
                continue
            tmp = ori_img[tly: tly + h, tlx: tlx + w, :]
            tmp2 = bin_clo[tly: tly + h, tlx: tlx + w]
            h, w = tmp.shape[:2]
            if h!=self.height or w!=self.width:
                dy = float(self.height) / h
                dx = float(self.width) / w

                tmp = cv2.resize(tmp, dsize=None, fx=dx, fy=dy, interpolation=cv2.INTER_LINEAR)
                tmp2 = cv2.resize(tmp2, dsize=None, fx=dx, fy=dy, interpolation=cv2.INTER_LINEAR)

            positive_patches.append(tmp)
            positive_gmask.append(tmp2)
            tmask[tly: tly + h, tlx: tlx + w] = 1.0
            # cv2.rectangle(ori_img, (tlx - 5, tly - 5), (tlx + w + 5, tly + h + 5), (255, 0, 0), 4)

        # get negative patches
        nidys, nidxs = np.where(tmask == 0.0)
        sample_num = 5000
        npos = [[y, x] for (y, x) in zip(nidys, nidxs)]
        npos_ids = np.random.choice([i for i in range(len(npos))], sample_num)
        # print(npos_ids)
        npos = np.array(npos)[npos_ids].tolist()
        for y, x in npos:
            # generate random box
            tlx = max(0, x - np.random.randint(10, 100)//2)
            tly = max(0, y - np.random.randint(10, 100)//2)
            rbx = min(tmask.shape[1], x + np.random.randint(10, 100)//2)
            rby = min(tmask.shape[0], y + np.random.randint(10, 100)//2)

            cmask = tmask[tly: rby, tlx: rbx]
            if cmask.sum() >= 0.2*(rby-tly)*(rbx-tlx):
                continue
            sgray_img = gray_im[tly: rby, tlx: rbx]
            if np.mean(sgray_img) < 135:
                continue

            tmp = ori_img[tly: rby, tlx: rbx, :]
            tmp2 = bin_clo[tly: rby, tlx: rbx]
            h, w = tmp.shape[:2]
            if h != self.height or w != self.width:
                dy = float(self.height) / h
                dx = float(self.width) / w

                tmp = cv2.resize(tmp, dsize=None, fx=dx, fy=dy, interpolation=cv2.INTER_LINEAR)
                tmp2 = cv2.resize(tmp2, dsize=None, fx=dx, fy=dy, interpolation=cv2.INTER_LINEAR)

            negative_patches.append(tmp)
            negative_gmask.append(tmp2)

        return positive_patches, positive_gmask, negative_patches, negative_gmask

    def load_img(self, image_name, other_image_name):
        positive_patches, positive_gmask, negative_patches, negative_gmask = self.load_patches(image_name)
        other_positive_patches, other_positive_gmask, other_negative_patches, other_negative_gmask = self.load_patches(other_image_name)

        assert len(positive_patches) > 0 or len(other_positive_patches) > 0
        for e1, e2 in zip(other_negative_patches, other_negative_gmask):
            positive_patches.append(e1)
            positive_gmask.append(e2)
        for e1, e2 in zip(other_negative_patches, other_negative_gmask):
            negative_patches.append(e1)
            negative_gmask.append(e2)

        return positive_patches, positive_gmask, negative_patches, negative_gmask


    def __len__(self):
        return self.items

    @staticmethod
    def map(image):  # normalize pixel value
        image = image / 255
        return image

    @staticmethod
    def unmap(image):
        image = image * 255
        image = np.clip(image, 0, 255).astype('uint8')
        return image


    def __call__(self, dataset_dict):
        # print("call cell dataset mapper...")
        dataset_dicts = {}

        data_index = dataset_dict['data_index']
        data_key = dataset_dict['key']

        # if self.data_name == 'monuseg':
        #     return self.load_data_monuseg(data_key)

        other_data_key = np.random.choice(self.img_lists, 1)[0]
        positive_patches, positive_gmask, negative_patches, negative_gmask = self.load_img(data_key, other_data_key)

        positive_imgs = positive_patches
        negative_imgs = negative_patches

        if self.transforms:

            positive_imgs = [np.array(self.transforms(Image.fromarray(positive_img))) for positive_img in positive_imgs]
            negative_imgs = [np.array(self.transforms(Image.fromarray(negative_img))) for negative_img in negative_imgs]

        # normalize image
        positive_imgs = [self.map(e) for e in positive_imgs]
        negative_imgs = [self.map(e) for e in negative_imgs]

        dataset_dicts['positive_imgs'] = torch.as_tensor(
            np.stack([
                np.ascontiguousarray(positive_img.transpose(2, 0, 1)) for positive_img in positive_imgs
            ])
        )  # [len, 3, H, W]
        dataset_dicts['negative_imgs'] = torch.as_tensor(
            np.stack([
                np.ascontiguousarray(negative_img.transpose(2, 0, 1)) for negative_img in negative_imgs
            ])
        )  # [len, 3, H, W]

        dataset_dicts['positive_gmasks'] = torch.as_tensor(np.stack(positive_gmask, axis=0))  # [len, H, W]
        dataset_dicts['negative_gmasks'] = torch.as_tensor(np.stack(negative_gmask, axis=0)) # [len, H, W]

        dataset_dicts["width"] = self.width
        dataset_dicts["height"] = self.height

        assert dataset_dicts['positive_imgs'].shape[0] == dataset_dicts['positive_gmasks'].shape[0]
        assert dataset_dicts['negative_imgs'].shape[0] == dataset_dicts['negative_gmasks'].shape[0]
        # assert 1 == 0
        return dataset_dicts

from torchvision.transforms import transforms
class GussianBlur(object):
    def __init__(self, kernel_size):

        radius = kernel_size // 2
        kernel_size = radius * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1), stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size), stride=1, padding=0, bias=False, groups=3)

        self.k = kernel_size
        self.r = radius
        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radius),
            self.blur_h,
            self.blur_v

        )
        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r+1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


