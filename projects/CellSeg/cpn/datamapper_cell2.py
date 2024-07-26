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
import matplotlib.pyplot as plt
from os.path import splitext


from .datasets.cpn import CPNTargetGenerator
from .datasets import cpn


class CellDatasetMapper(DatasetMapper):
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

        if not self.is_cos:
            legal_img_list = pickle.load(
                open(
                    "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/合并标注数据/data_512_new/legal_" + self.split + "_files.pkl",
                    'rb')
            )
            clegal_img_list = []
            for e in legal_img_list:
                if e == 'dfa69dce-d7e7-11ec-a8b3-d562838f25df_2.bmp':
                    # print("dddd")
                    # print(e)
                    # assert 1 == 0
                    e = splitext(e)[0] + '.png'
                # else:
                clegal_img_list.append(e)

            self.legal_img_list = legal_img_list

        else:
            img_files = "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/new_seg_data/jzx/"
            # img_files = "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/single_test"


        self.items = len(self.legal_img_list)

        self.gen = CPNTargetGenerator(
            samples=cfg.samples,
            order=cfg.order,
            min_fg_dist=cfg.min_fg_dist,
            max_bg_dist=cfg.max_bg_dist
        )
        self.transforms = None if cfg.transforms == 0 else cfg.transforms
        self.data_file = cfg.data_file
        self.img_root = cfg.img_file
        self.mask_root = cfg.mask_root
        self.height = 512
        self.width = 512

        self.data_name = cfg.data_name
        # if not self.is_cos and not (self.data_name in ['dsb'] and self.split=='test'):
        self.preload()


    def load_data_for_cos(self, file_root, img_name):
        print("#################")
        print(img_name)
        ori_img = cv2.imread(os.path.join(file_root, img_name), 3)  # BGR

        height, width = ori_img.shape[:2]
        if height != 512 or width != 512:

            ramdon = False
            if ramdon:
                print("random clip (" + str(height) + ", " + str(width) + ") to (512, 512)")
                start_x = random.randint(0, height - 512)
                start_y = random.randint(0, width - 512)
                ori_img = ori_img[start_x: start_x + 512, start_y: start_y + 512, :]
            else: # resize
                print("resize (" + str(height) + ", " + str(width) + ") to (512, 512)")
                dy = float(self.height)/height
                dx = float(self.width)/width

                ori_img = cv2.resize(ori_img, dsize=None, fx=dx, fy=dy, interpolation=cv2.INTER_LINEAR)


            # cv2.imshow('ori img', ori_img)
            # cv2.waitKey(0)

        # continue

        height, width = ori_img.shape[:2]
        assert height == self.height and width == self.width, ori_img.shape
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)  # convert to RGB

        return {
            'image_name': img_name,
            'image': ori_img,
            'height': self.height,
            'width': self.width,

        }

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

    def preload(self):
        oriimages = {}
        orimasks = {}
        for imgname in os.listdir(self.img_root):
            ori_im = cv2.imread(os.path.join(self.img_root, imgname), 3)
            ori_im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            ori_mask = cv2.imread(os.path.join(self.mask_root, splitext(imgname)[0]+'.png'), 3)
            ori_mask = cv2.cvtColor(ori_mask, cv2.COLOR_BGR2GRAY)

            oriimages[imgname] = ori_im
            orimasks[imgname] = ori_mask
        self.oriimages = oriimages
        self.orimasks = orimasks


    def datagenerate(self, imgname, cbox, contours):
        assert imgname in self.oriimages
        ori_im = self.oriimages[imgname]  # RGB
        cx, cy, cw, ch = cbox
        im = ori_im[cy: cy+ch, cx: cx+cw, :] # [H, W, 3]

        ori_mask = self.orimasks[imgname]
        mask = ori_mask[cy: cy+ch, cx: cx+cw] # GRAY
        mask[mask < 120] = 0  # [0, 1]
        mask[mask != 0] = 1
        assert len(np.unique(mask).tolist()) <= 2, np.unique(mask)

        assert im.shape[0] == self.height and im.shape[1] == self.width
        contours_ = []
        print("number of contours: ", len(contours))
        for el in contours:
            # print(type(el))
            # print(np.array(el).squeeze(1).shape)
            contours_.append(np.array(el).squeeze(1))  # [num_points, 1, 2] -> [num_points, 2]

        labels = cpn.contours2labels(contours_, (self.height, self.width))
        labels_ = cpn.resolve_label_channels(labels)

        return {
                'image_name': imgname,
                'image': im,
                'labels': labels_,
                'height': self.height,
                'width': self.width,
                'bmask': mask,
        }

    def generate_freq_guild(self, gray_img):

        dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
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
        low_freq_guild = res_low / float(np.max(res_low))  # [H, W]

        return low_freq_guild


    def __call__(self, dataset_dict):
        # print("call cell dataset mapper...")
        dataset_dicts = {}

        data_index = dataset_dict['data_index']
        data_key = dataset_dict['key']

        # if self.data_name == 'monuseg':
        #     return self.load_data_monuseg(data_key)

        if self.is_cos:
            file_root = dataset_dict['file_root']
            dataset = self.load_data_for_cos(file_root, data_key)
            img = dataset['image']

            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            if self.use_freq_guild:
                low_freq_guild = self.generate_freq_guild(gray_img)

            fgimg_mask = gray_img < 200  # black: True, white: False

            # normalize image
            img = self.map(img)

            dataset_dicts["width"] = self.width
            dataset_dicts["height"] = self.height

            dataset_dicts['image'] = torch.as_tensor(
                np.ascontiguousarray(img.transpose(2, 0, 1))
            )  # [3, H, W]

            dataset_dicts['file_name'] = dataset['image_name']

            dataset_dicts['fgimg_mask'] = fgimg_mask.astype('uint8')  # [H, W] # 1 for black, 0 for white
            # assert 1 == 0
            if self.use_freq_guild:
                dataset_dicts['low_freq_guild'] = low_freq_guild

            return dataset_dicts

        contours = dataset_dict['contours']
        cbox = dataset_dict['cbox'] # x, y, w, h

        dataset = self.datagenerate(data_key, cbox, contours)
        img = dataset['image']
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if self.use_freq_guild:
            low_freq_guild = self.generate_freq_guild(gray_img)

        labels = dataset['labels']

        # generate contours and fourier
        gen = self.gen
        gen.feed(labels=labels)

        labels_ = gen.reduced_labels

        if self.transforms:
            pass

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # normalize image
        img = self.map(img)

        dataset_dicts["width"] = self.width
        dataset_dicts["height"] = self.height

        dataset_dicts['image'] = torch.as_tensor(
            np.ascontiguousarray(img.transpose(2, 0, 1))
        )  # [3, H, W]
        dataset_dicts['bmask'] = dataset['bmask'] # [H, W]
        dataset_dicts['labels'] = labels_  # [H, W]
        dataset_dicts['fouriers'] = (gen.fourier.astype('float32'), )
        dataset_dicts['locations'] = (gen.locations.astype('float32'), )
        dataset_dicts['sampled_contours'] = (gen.sampled_contours.astype('float32'), )
        dataset_dicts['sampling'] = (gen.sampling.astype('float32'), )
        dataset_dicts['file_name'] = dataset['image_name']

        if self.use_freq_guild:
            dataset_dicts['low_freq_guild'] = low_freq_guild.astype('float32')  # [0, 1], float

        # for k in dataset_dicts.keys():
        #     v = dataset_dicts[k]
        #     print(k, end=" ")
        #     if isinstance(v, str):
        #         print(v)
        #     elif isinstance(v, int):
        #         print(v)
        #     else:
        #         print(v.shape)
        # image
        # torch.Size([3, 512, 512])
        # labels
        # torch.Size([512, 512])
        # fouriers
        # torch.Size([3, 25, 4])
        # locations
        # torch.Size([3, 2])
        # sampled_contours
        # torch.Size([3, 128, 2])
        # sampling
        # torch.Size([128])

        # assert 1 == 0
        return dataset_dicts





