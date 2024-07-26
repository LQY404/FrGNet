import os.path as osp
import json
import os
import random

from detectron2.data import DatasetCatalog, MetadataCatalog
import pickle
import numpy as np
import scipy
import pickle


def register_cell(split='train'):
    anno_root = "/home/kyfq/data/MoNuSeg 2018 Training Data/Data_mrcnn"

    data_dict = []
    for idx in range(len(os.listdir(anno_root))):
        data_dict.append({
            'data_index': idx,
        })
    if split == 'train':
        random.shuffle(data_dict)
    # if len(data_dict) == 0:
    #     data_dict.append({
    #         'image_id': 142,
    #         'ref_id': 0,
    #         'raw_sent': ""
    #     })

    print("data for " + split + ": ", len(data_dict))
    # assert 1 == 0
    return data_dict


def register_cell_all():
    print("注册cell数据集")

    split = 'train'
    # split = 'test'
    # split = 'cos'

    if split == 'train':

        DatasetCatalog.register("cell_" + "train", lambda: register_cell())
        MetadataCatalog.get('cell_' + "train").set(json_file=None, evaluator_type="refcoco", thing_classes=["cell"])
    
    elif split == 'test':
    # split = 'test'

        DatasetCatalog.register("cell_" + 'test', lambda: register_cell(split=split))
        MetadataCatalog.get('cell_' + 'test').set(json_file=None, evaluator_type="refcoco", thing_classes=["cell"])

    elif split == 'cos':
        DatasetCatalog.register("cell_" + 'test', lambda: register_cell_cos())
        MetadataCatalog.get('cell_' + 'test').set(json_file=None, evaluator_type="refcoco", thing_classes=["cell"])

    else:
        raise()

register_cell_all()