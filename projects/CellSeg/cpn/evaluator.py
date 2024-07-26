import contextlib
import copy
from detectron2.structures.masks import BitMasks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
import io
import itertools
import json
import logging
import numpy as np
import os
import re
import torch
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils.visualizer import Visualizer, GenericMask, ColorMode

import glob
import shutil
import zipfile

import matplotlib.pyplot as plt

from .datasets.instance_eval import LabelMatcherList, LabelMatcher

class CellEvaluator(DatasetEvaluator):
    def __init__(self, cfg=None):
        self._cpu_device = torch.device("cpu")
        self.cfg = cfg

        self.mask_ious = []
        # self.save_root = "./inference_img"
        # os.makedirs(self.save_root, exist_ok=True)
        self.lml = []


    def process(self, inputs, outputs):

        # 准备工作
        # return {}

        for input, output in zip(inputs, outputs):

            self.lml.append([])

    def compute_dice_ins(self, preds, gts):
        #
        pass

    def compute_aji(self, true, pred): # aggregated jaccard index
        true = np.copy(true)
        pred = np.copy(pred)
        true_id_list = list(np.unique(true))
        pred_id_list = list(np.unique(pred))

        if len(pred_id_list) == 1:
            return 0

        true_masks = [None, ]
        for t in true_id_list[1: ]:
            t_mask = np.array(true == t, np.uint8)
            true_masks.append(t_mask)

        pred_masks = [None, ]
        for p in pred_id_list[1:]:
            p_mask = np.array(pred == p, np.uint8)
            pred_masks.append(p_mask)

        pairwise_inter = np.zeros(
            [len(true_id_list)-1, len(pred_id_list)-1], dtype=np.float64
        )
        pairwise_union = np.zeros(
            [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
        )

        for true_id in true_id_list[1: ]: # 0-th is background
            t_mask = true_masks[true_id]
            pred_true_overlap = pred[t_mask > 0]
            pred_true_overlap_id = np.unique(pred_true_overlap)
            pred_true_overlap_id = list(pred_true_overlap_id)
            for pred_id in pred_true_overlap_id:
                if pred_id == 0: # background, ignore
                    continue
                p_mask = pred_masks[pred_id]
                total = (t_mask + p_mask).sum()
                inter = (t_mask * p_mask).sum()
                pairwise_inter[true_id-1, pred_id-1] = inter
                pairwise_union[true_id-1, pred_id-1] = total - inter

        pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
        paired_pred = np.argmax(pairwise_iou, axis=1)
        pairwise_iou = np.max(pairwise_iou, axis=1)

        paired_true = np.nonzero(pairwise_iou > 0.0)[0]
        paired_pred = paired_pred[paired_true]

        overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
        overall_union = (pairwise_union[paired_true, paired_pred]).sum()

        paired_true = list(paired_true + 1) # index to instance id
        paired_pred = list(paired_pred + 1)

        unpaired_pred = np.array(
            [idx for idx in pred_id_list[1: ] if idx not in paired_pred]
        )
        unpaired_true = np.array(
            [idx for idx in true_id_list[1:] if idx not in paired_true]
        )

        for true_id in unpaired_true:
            overall_union += true_masks[true_id].sum()
        for pred_id in unpaired_pred:
            overall_union += pred_masks[pred_id].sum()

        aji_score = overall_inter / overall_union

        return aji_score




def evaluate(self):
        pass