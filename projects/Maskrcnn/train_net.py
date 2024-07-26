#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SparseRCNN Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

from operator import is_
import os
import itertools
# import time
from typing import Any, Dict, List, Set
import numpy as np
import torch
import random

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import AutogradProfiler, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

# set random seed
seed = 53717220
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# from mrcnnref import add_mrcnnref_config, RefcocoEvaluator, \
# IEPDatasetMapperWithBasis, SketchDatasetMapper, PhraseCutDatasetMapper, RefcocoSingleDatasetMapper
# from mrcnnref import add_mrcnnref_config, SketchDatasetMapper, RefcocoEvaluator, PhraseCutDatasetMapper, RefcocoSingleDatasetMapper

from mrcnn import add_cell_config, CellDatasetMapper

class Trainer(DefaultTrainer):
#     """
#     Extension of the Trainer class adapted to SparseRCNN.
#     """

    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # 最好再重写一下
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)

        if evaluator_type == "refcoco":
            # return RefcocoEvaluator(cfg)
            return CellEvaluator(cfg)
            # assert 1 == 0
            # return None
        
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


    @classmethod
    def build_train_loader(cls, cfg):


        if cfg.MODEL.NUM_CLASSES == 1:
            print("cell")
            mapper = CellDatasetMapper(cfg, is_train=True)

        else:
            raise 

        return build_detection_train_loader(cfg, mapper=mapper, num_workers=4)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        print("使用重写的test_dataloader")
        # mapper = IEPDatasetMapperWithBasis(cfg, False)
        # mapper = ReferringDatasetMapperWithBasis(cfg, False)

        if cfg.MODEL.NUM_CLASSES == 1:
            print("cell")
            mapper = CellDatasetMapper(cfg, is_train=False)
        else:
            raise 

        dataset_name = cfg.DATASETS.TEST[0]
        return build_detection_test_loader(cfg, mapper=mapper, num_workers=1, dataset_name=dataset_name)


    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        
        # 使用预训练模型时这个地方需要修改
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                print("不需要训练部分,", key)
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            
            # if "backbone" in key:
            if "backbone" in key: # 不微调FPN
                print("微调部分", key)
                # lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER  # fine-tune backbone
            
            elif "bert" in key:
                if "encoder.layer" not in key:
                    continue
                
                print("微调：", key)
                # lr = lr * cfg.SOLVER.TEXTENCODER  # fine-tune bert
            elif "clip" in key:
                if "encoder.layer" not in key:
                    continue
                
                print("微调：", key)
                # lr = lr * cfg.SOLVER.TEXTENCODER  # fine-tune clip
                  
            else:
                print("不微调,", key)
                
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            print(key, lr)
            # print(key)
            # 包括：backbone, proposal_generator, roi_heads, textencoder
        # print(params)
        # for p in params:
        #     print(p['lr'])
        # assert 1 == 0
        

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)

        
        # print(optimizer)

        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    print("read config from given config file...")
    # cfg.merge_from_file(args.config_file)
    print("merge config from cmd")
    cfg.merge_from_list(args.opts)
    print("add config from given config file")
    add_cell_config(cfg)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        # print(model)
        # assert 1 == 0
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    # print(cfg)
    # assert 1 == 0
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    # assert 1 == 0
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
