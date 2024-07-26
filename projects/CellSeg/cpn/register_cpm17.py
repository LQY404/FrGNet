import os.path as osp
import json
import os
import random

from detectron2.data import DatasetCatalog, MetadataCatalog
import pickle
import numpy as np
import scipy
import pickle


def register_cell_cos():
    # excimg = os.listdir("/home/kyfq/detectron2/inferencevis_1712718119.7641475")
    excimg = []

    img_files = "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/new_seg_data/jzx/"
    # img_files = "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/single_test"
    all_files = os.listdir(img_files)  # ori_imgs
    cimg_names = []
    for e in all_files:
        if e.split(".")[-1] not in ['png', 'bmp', 'jpg']:
            # print(e)
            continue

        f = False
        for ex in excimg:
            if e in ex:
                f = True
                break
        if f:
            print("#################### already inference img: ", e)
            continue

        cimg_names.append(e)
    # print(cimg_names)
    # assert 1 == 0
    data_dict = []
    for i in range(len(cimg_names)):
        file_name = cimg_names[i]
        data_dict.append({
            'data_index': i,
            'key': file_name,
            'file_root': img_files
        })

    img_files = "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/new_seg_data/tct/"
    all_files = os.listdir(img_files)  # ori_imgs
    cimg_names = []
    for e in all_files:
        if e.split(".")[-1] not in ['png', 'bmp', 'jpg']:
            # print(e)
            continue

        f = False
        for ex in excimg:
            if e in ex:
                f = True
                break
        if f:
            print("#################### already inference img: ", e)
            continue

        cimg_names.append(e)
    for i in range(len(cimg_names)):
        file_name = cimg_names[i]
        data_dict.append({
            'data_index': i,
            'key': file_name,
            'file_root': img_files
        })
    # print(data_dict)
    # assert 1 == 0
    return data_dict


def register_cell(split='train'):
    # assert 1 == 0
    print("注册", split)
    json_file = "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/合并标注数据/data_512_new/annos.json"

    if split == 'train':
        img_files = "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/cpm17/train/Images_512"
    else:
        img_files = "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/cpm17/test/Images_512"
    all_files = os.listdir(img_files) # ori_imgs
    # if split == 'train':
    #     all_files = all_files[: int(len(all_files)*0.7)]
    # else:
    #     all_files = all_files[int(len(all_files)*0.7): ]

    # assert 1 == 0
    data_dict = []

    for i in range(len(all_files)):
        file_name = all_files[i]

        data_dict.append({
            'data_index': i,
            'key': file_name,
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
    print("注册cpm17数据集")

    split = 'train'
    split = 'test'
    # split = 'cos'


    if split == 'train':

        DatasetCatalog.register("cpm17_" + "train", lambda: register_cell())
        MetadataCatalog.get('cpm17_' + "train").set(json_file=None, evaluator_type="refcoco", thing_classes=["cell"])

    elif split == 'test':
        # split = 'test'

        DatasetCatalog.register("cpm17_" + 'test', lambda: register_cell(split=split))
        MetadataCatalog.get('cpm17_' + 'test').set(json_file=None, evaluator_type="refcoco", thing_classes=["cell"])

    elif split == 'cos':
        DatasetCatalog.register("cpm17_" + 'test', lambda: register_cell_cos())
        MetadataCatalog.get('cpm17_' + 'test').set(json_file=None, evaluator_type="refcoco", thing_classes=["cell"])


    else:
        raise ()


register_cell_all()