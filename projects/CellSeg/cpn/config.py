# -*- coding: utf-8 -*-
#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_cell_config(cfg):
    """
    Add config for Mask R-CNN REF.
    """


    # samples, order, max_bg_dist, min_fg_dist, transforms = None, is_train = True, img_file = None, data_file = None, items = 2 ** 12
    cfg.samples = 224 # number of coordinates per contour
    cfg.order = 25 # the higher, the more complex shapes can be detected
    cfg.max_bg_dist = 0.8
    cfg.min_fg_dist = 0.85
    # cfg.bg_fg_dists = (0.8, 0.85)
    cfg.split = 'test'
    cfg.data_name = ['our', 'monuseg', 'dsb', 'tnbc', 'cpm17'][1]
    cfg.mask_root = {
        'our': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/合并标注数据/data_512_new/mask/",
        'monuseg': {
            'train': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/MoNuSeg 2018 Training Data/MoNuSeg 2018 Training Data/mask_all/",
            'test': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/MoNuSegTestData/MoNuSegTestData/mask_all"
        }[cfg.split],
        'dsb': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/DSB/stage1_train_512/masks_all",
        'tnbc': {
            'train': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/TNBC_NucleiSegmentation/mask_all",
            'test': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/TNBC_NucleiSegmentation/mask_all_test"
        }[cfg.split],
        'cpm17': {
            'train': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/cpm17/train/masks_all/",
            'test': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/cpm17/test/masks_all/",
        }[cfg.split],
    }[cfg.data_name]
    cfg.img_file = {
        'our': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/合并标注数据/data_512_new/img/",
        'monuseg': {
            'train': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/MoNuSeg 2018 Training Data/MoNuSeg 2018 Training Data/Tissue Images_512/",
            'test': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/MoNuSegTestData/MoNuSegTestData/Tissue Images_512"
        }[cfg.split],
        'dsb': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/DSB/stage1_train_512/images",
        'tnbc': {
            'train': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/TNBC_NucleiSegmentation/images_512",
            'test': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/TNBC_NucleiSegmentation/image_test_512"
        }[cfg.split],
        'cpm17': {
            'train': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/cpm17/train/Images_512/",
            'test': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/cpm17/test/Images_512/",
        }[cfg.split],
    }[cfg.data_name]
    cfg.data_file = {
        'our': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/合并标注数据/data_512_new/annos.json",
        'monuseg': {
            'train': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/MoNuSeg 2018 Training Data/MoNuSeg 2018 Training Data/anno_train.json",
            'test': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/MoNuSegTestData/MoNuSegTestData/anno_test.json"
        }[cfg.split],
        'dsb': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/DSB/anno_train.json",
        'tnbc': {
            'train': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/TNBC_NucleiSegmentation/anno_train.json",
            'test': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/TNBC_NucleiSegmentation/anno_test.json"
        }[cfg.split],
        'cpm17': {
            'train': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/cpm17/train/anno_train.json",
            'test': "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/cpm17/test/anno_test.json"
        }[cfg.split],
    }[cfg.data_name]

    cfg.transforms = 0
    # cfg.item = 4500 # can't use the name "items"
    cfg.item = {
        'our': [4500, 3500][1], # 4500 for ori image, 3500 for rimage,
        'monuseg': 37,
        'dsb': 670,
        'tnbc': 35,
        'cpm17': 32
    }[cfg.data_name]

    cfg.DATASETS.TRAIN = {
        'our': ("cell_train",),
        'monuseg': ("monuseg_train",),
        'dsb': ("dsb_train",),
        'tnbc': ("tnbc_train",),
        'cpm17': ("cpm17_train",)
    }[cfg.data_name]
    cfg.DATASETS.TEST = {
        'our': ("cell_test",),
        'monuseg': ("monuseg_test",),
        'dsb': ("dsb_test",),
        'tnbc': ("tnbc_test",),
        'cpm17': ("cpm17_test",)
    }[cfg.data_name]

    cfg.is_cos = True
    cfg.use_freq_guild = 'seg' # ['concate', 'seg', None]
    cfg.ignore_bg = False
    cfg.use_topo_loss = False
    cfg.fusion_type = 'dattn' # ['attn', 'fusion', 'dattn']
    cfg.use_contr = False


    print("add extra config...")
    cfg.MODEL.META_ARCHITECTURE = 'CPNModel'
    # cfg.MODEL.NUM_CLASSES = 35  # cad
    cfg.MODEL.NUM_CLASSES = 1

    # cfg.local_rank = 3
    cfg.SOLVER.IMS_PER_BATCH = 2
    if cfg.MODEL.NUM_CLASSES == 35 or cfg.MODEL.NUM_CLASSES == 1:
        print("sketch模式不需要冻结resnet")
        # cfg.MODEL.BACKBONE.NAME = "build_resnet_sketch_fpn_backbone"

        # cfg.MODEL.WEIGHTS = "/nfs/TEMP/pretrained_mrcnn/model_final.pth"  # dilation_v1, 512
        # cfg.MODEL.WEIGHTS = "/nfs/TEMP/pretrained_mrcnn_dilation_v1_768/model_final.pth"  #dilation_v1 768

        # cfg.MODEL.WEIGHTS = "/nfs/TEMP/pretrained_mrcnn_dilation_v2/model_final.pth"  # dilation_v2

        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        # cfg.DATASETS.TRAIN = ("cad_train", )
        # cfg.DATASETS.TEST = ('cad_test', )
        # cfg.DATASETS.TRAIN = ("cell_train",)
        # cfg.DATASETS.TEST = ('cell_test',)
        # cfg.SOLVER.MAX_ITER = 360000
        # cfg.SOLVER.STEPS = (270000, 330000)
        # cfg.SOLVER.MAX_ITER = 510000
        # cfg.SOLVER.STEPS = (390000, 480000)
        # cfg.DATA_NUM = 30000
        # cfg.all_iter = 3000000  # 30epoch
        # cfg.all_iter = 3000000  # 50epoch
        # cfg.all_iter = 30000*100  # 100 epoch，没有做重采样
        # cfg.all_iter = 30000*100  # 100 epoch，没有做重采样

        cfg.all_iter = cfg.item * 30  # 如果使用 oriconfig，根据对loss以及训练输出的分析，认为epoch可以降低到50-80，先降低到70试试

        # cfg.all_iter = 1 * 800
        # cfg.all_iter = 10 * 1000

        # cfg.all_iter = 15000*100

        # cfg.SOLVER.WARMUP_ITERS = int(cfg.all_iter * 0.2)
        # cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
        cfg.SOLVER.MAX_ITER = cfg.all_iter // cfg.SOLVER.IMS_PER_BATCH
        cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.MAX_ITER * 0.1)
        cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
        # cfg.SOLVER.GAMMA = cfg.SOLVER.WARMUP_FACTOR
        cfg.SOLVER.STEPS = (int(cfg.all_iter * 0.8) // min(16, cfg.SOLVER.IMS_PER_BATCH),
                            int(cfg.all_iter * 0.9) // min(16, cfg.SOLVER.IMS_PER_BATCH))
        # cfg.SOLVER.STEPS = (int(cfg.all_iter * 1.0) // min(8, cfg.SOLVER.IMS_PER_BATCH), int(cfg.all_iter * 1.0) // min(8, cfg.SOLVER.IMS_PER_BATCH))

    cfg.VIS_PERIOD = -1  # 设置太小会导致模型占据CPU内存过多

    cfg.e_sche = [[0, 0.1 * cfg.all_iter], [0.4 * cfg.all_iter, 0.5 * cfg.all_iter],
                  [0.6 * cfg.all_iter, 0.65 * cfg.all_iter]]

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    # cfg.SOLVER.OPTIMIZER = 'SGD'
    # cfg.SOLVER.BACKBONE_MULTIPLIER = 0.01
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    # cfg.SOLVER.TEXTENCODER = 0.0001
    # cfg.SOLVER.BASE_LR = 0.00005
    # cfg.SOLVER.BASE_LR = 0.00002
    cfg.SOLVER.BASE_LR = 0.0005 # 单个可用
    # cfg.SOLVER.BASE_LR = 0.0002  # 20个可用
    # cfg.SOLVER.BASE_LR = 0.0005
    # cfg.SOLVER.BASE_LR = 0.00025
    # cfg.SOLVER.BASE_LR = (0.00025/8) * cfg.SOLVER.IMS_PER_BATCH # 保持bs/lr为定值
    # cfg.SOLVER.BASE_LR = 0.001 * max(1.0, (cfg.SOLVER.IMS_PER_BATCH / 8))