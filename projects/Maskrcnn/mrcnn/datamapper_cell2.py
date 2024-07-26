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

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)


class CellDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        print("for cell")
        # print(cfg.all_iter, cfg.e_sche, cfg.samples, cfg.img_file, cfg.data_file, cfg.transforms, cfg.item)
        # Rebuild augmentations
        # print(cfg)
        # print(cfg.items)
        # self.items = None if cfg.item <= 0 else cfg.item
        # print(self.items)
        # assert 1 == 0
        self.split = 'train' if is_train else 'test'
        # self.split = 'train'
        # self.split = 'val'
        # self.split = 'train'
        self.training = is_train

        # self.is_cos = cfg.is_cos
        # self.use_freq_guild = cfg.use_freq_guild

        print("训练模式") if is_train else print("测试模式")

        self.transforms = None if cfg.transforms == 0 else cfg.transforms
        # self.data_file = cfg.data_file
        # self.img_root = cfg.img_file
        # self.mask_root = cfg.mask_root
        self.height = 384
        self.width = 384

        self.data_name = cfg.data_name
        # if not self.is_cos and not (self.data_name in ['dsb'] and self.split=='test'):
        self.preload()

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
        anno_root = "/home/kyfq/data/MoNuSeg 2018 Training Data/Data_mrcnn"
        self.items = len(os.listdir((anno_root)))
        ins_annos = {}
        for i, anno_file in enumerate(os.listdir(anno_root)):
            ins_anno = np.load(os.path.join(anno_root, anno_file), allow_pickle=True).item()
            # print(ins_anno.bboxes)
            # print(type(ins_anno))
            # print(ins_anno.keys()) # dict_keys(['image_name', 'bmasks', 'bboxes', 'labels'])
            # assert 1 == 0
            ins_annos[i] = ins_anno

        assert len(ins_annos) == self.items
        self.ins_annos = ins_annos
        # self.orimasks = orimasks


    def __call__(self, dataset_dict):
        # print("call cell dataset mapper...")
        dataset_dicts = {}

        data_index = dataset_dict['data_index']
        datas = self.ins_annos[data_index]
        # print(datas.keys())

        img_name = datas['image_name']
        image = datas['image'] # [H, W, 3]
        labels = datas['labels']
        bboxes = datas['bboxes']
        bmasks = datas['bmasks']

        bboxes = np.array(bboxes) # [N_ins, 4]
        targets = Instances((self.height, self.width))
        targets.gt_boxes = Boxes(list(bboxes))
        classes = torch.tensor(labels, dtype=torch.int64)
        targets.gt_classes = classes

        # print(targets)
        # assert 1 == 0

        masks = BitMasks(torch.stack(
            [torch.from_numpy(np.ascontiguousarray(e)) for e in bmasks]
        ))
        targets.gt_masks = masks

        dataset_dicts['image'] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        dataset_dicts['image_name'] = img_name
        dataset_dicts['height'] = self.height
        dataset_dicts['width'] = self.width
        dataset_dicts['instances'] = utils.filter_empty_instances(targets)

        # assert 1 == 0
        return dataset_dicts





