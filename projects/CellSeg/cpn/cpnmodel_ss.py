# -*- coding: utf-8 -*-
import math

# from projects.FloorCAD.floorcad.gat import LayerType
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Sequential as Seq
from torchvision import transforms as trans
import cv2
from scipy.optimize import linear_sum_assignment
from hausdorff import hausdorff_distance

from detectron2.structures import ImageList, Instances
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from detectron2.config import configurable
from .model import cpn
from .datasets.misc import universal_dict_collate_fn
from .util import util
from .datamapper_cell import CellDatasetMapper
from .visualization import images as vis

@META_ARCH_REGISTRY.register()
class CPNModel_SS(torch.nn.Module):

    @configurable
    def __init__(self,
                 *,
                 pixel_mean: Tuple[float],
                 pixel_std: Tuple[float],
                 input_format: Optional[str] = None,
                 vis_period: int = 0,
                 cfg=None,
                 ):
        super().__init__()

        self.num_classes = cfg.MODEL.NUM_CLASSES if cfg else 1

        self.input_format = input_format
        self.vis_period = cfg.VIS_PERIOD
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"


        self.time = 1

        self.use_freq_guild = cfg.use_freq_guild  # ['concate', 'seg', None]
        assert self.use_freq_guild in ['concate', 'seg', None]

        model_weights = torch.load('../instance_seg/ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c.pt')
        # state_dict = model_weights['state_dict']
        conf = model_weights['cd.models']
        kw = {**conf.get('kwargs', conf.get('kw', {}))}
        # print("conf: ") # nothing, default, set using kw
        # print(*conf.get('args', conf.get('a', ())))
        # print("kw: ")
        # print(kw)
        # {'backbone_kwargs': {'inputs_mean': [0.485, 0.456, 0.406], 'inputs_std': [0.229, 0.224, 0.225],
        #                      'bridge_strides': False, 'block_cls': 'ResBlock',
        #                      'block_kwargs': {'activation': 'LeakyReLU',
        #                                       'norm_layer': {'NormProxy': {'norm': 'GroupNorm', 'num_groups': 32}}},
        #                      'backbone_kwargs': {'pyramid_pooling': True, 'pyramid_pooling_channels': 64,
        #                                          'pyramid_pooling_kwargs': {'method': 'ppm', 'concatenate': False}}},
        #  'certainty_thresh': None, 'classes': 2, 'contour_head_stride': 1, 'in_channels': 3,
        #  'nms_thresh': 0.3141592653589793, 'order': 25, 'order_weights': True, 'pretrained': False, 'refinement': True,
        #  'refinement_buckets': 6, 'refinement_head_stride': 1, 'refinement_interpolation': 'bilinear',
        #  'refinement_iterations': 4, 'refinement_margin': 3.0, 'samples': 224, 'score_thresh': 0.9,
        #  'uncertainty_factor': 7.0, 'uncertainty_head': True, 'uncertainty_nms': True,
        #  'score_features': ('1', 'encoder.1'), 'contour_features': '1', 'fourier_features': '1',
        #  'location_features': '1', 'refinement_features': ('0', 'encoder.0'), 'uncertainty_features': '1',
        #  'fuse_kwargs': {'norm_layer': None, 'activation': None, 'bias': False}}
        # assert 1 == 0

        # self.fre_guild = conf.use_guild
        # if self.fre_guild:
        #     kw[]
        if self.use_freq_guild is not None:
            print("use low frequency img as guild...")
            # assert 1 == 0
            cstate_dict = {}
            # for k in state_dict:
            #     print(k)
            #     if 'core.backbone.body.0.0' in k:
            #         print('remove layer: ', k)
            #         continue
            #     cstate_dict[k] = state_dict[k]
            # assert 1 == 0

            # if we just concat the img and guild as input, set below as false
            if self.use_freq_guild == 'seg':
                kw['in_channels'] = 3

            elif self.use_freq_guild == 'concate':
                kw['in_channels'] = 4

            else:
                raise

        kw['use_guild'] = self.use_freq_guild
        kw['backbone_kwargs']['use_guild'] = self.use_freq_guild

        kw['ignore_bg'] = cfg.ignore_bg
        kw['backbone_kwargs']['ignore_bg'] = cfg.ignore_bg

        kw['use_topo_loss'] = cfg.use_topo_loss
        kw['backbone_kwargs']['use_topo_loss'] = cfg.use_topo_loss
        # kw['back']

        kw['fusion_type'] = cfg.fusion_type
        kw['backbone_kwargs']['fusion_type'] = cfg.fusion_type

        # kw['guild_features'] = ('0', '1', '2')
        kw['use_contr'] = cfg.use_contr

        change_feature = False
        if change_feature:
            print("change_feature...")
            kw['score_features'] = ('0')
            kw['contour_features'] = '0'
            kw['fourier_features'] = '0'
            kw['location_features'] = '0'
            kw['uncertainty_features'] = '0'


        # model = cpn.CpnResNeXt101UNet(*conf.get('args', conf.get('a', ())), **kw)
        # model = cpn.CpnResNeXt50UNet(*conf.get('args', conf.get('a', ())), **kw)
        model = cpn.CpnResNet101UNet(*conf.get('args', conf.get('a', ())), **kw)
        # model = cpn.CpnResNet50UNet(*conf.get('args', conf.get('a', ())), **kw)
        # kw['backbone_kwargs']['backbone_kwargs'] = None
        # model = cpn.CpnU22(*conf.get('args', conf.get('a', ())), **kw)
        # kw['refinement_features'] = ('0')
        # kw['score_features'] = ('1')
        # model = cpn.CpnResNet101FPN(*conf.get('args', conf.get('a', ())), **kw)

        # print(model)
        # x = torch.randn((2, 3, 512, 512)).to(self.device)
        # model(x)
        # assert 1 == 0
        # print(state_dict.keys())

        # model_seg = cpn2.CpnResNet101UNet(*conf.get('args', conf.get('a', ())), **kw)
        # model_seg.to(self.device)
        # self.model_seg = model_seg
        self.model_seg = None

        # print(model)
        # assert 1 == 0
        # if not change_feature:
        #     model.load_state_dict(state_dict if not self.use_freq_guild else cstate_dict, strict=False)

        model.to(self.device)

        self.model = model
        # assert 1 == 0

        # self.vis_file = "./trainvis_"+str(time.time())
        self.vis_file = "./inferencevis_" + str(time.time())
        os.makedirs(self.vis_file, exist_ok=True)
        self.compute_metric = False
        self.ajis = []
        self.dq = []
        self.sq = []
        self.pq = []



    @classmethod
    def from_config(cls, cfg):

        return {
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "cfg": cfg
        }

    @property
    def device(self):
        return self.pixel_mean.device

    # def _init_weights(self):

    def normalize_input(self, inputs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        assert_range = (0., 1.)
        transform = trans.Compose([
            trans.Normalize(mean=mean, std=std)
        ])
        if assert_range is not None:
            assert torch.all(inputs >= assert_range[0]) and torch.all(
                inputs <= assert_range[1]), f'Inputs should be in interval {assert_range}'

        inputs = transform(inputs)
        return inputs

    def preprocess_image(self, images_list):
        """
        Normalize, pad and batch the input images.
        """
        images = images_list
        # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # if self.use_freq_guild is not None:
        images = [self.normalize_input(e) for e in images]

        images = ImageList.from_tensors(images, 0)

        return images

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):

        # self.time += 1
        if not self.training:
            return self.inference(batched_inputs)

        assert not torch.jit.is_scripting(), "Scripting for training mode is not supported."
        # print(len(batched_inputs))
        #
        # assert len(batched_inputs) == 1  # 仅支持1batch size，使用多卡即可
        self.time += 1
        vis = False
        if self.time>0 and self.time%self.vis_period==0:
            print("#"*20)
            print("visualize during training...")
            self.inference(batched_inputs)
            self.model.train()
            print("#" * 20)
            vis = True
            # return

        pimages = self.preprocess_image([x["pimage"].to(self.device) for x in batched_inputs]).tensor.to(
            torch.float32)  # [B, 3, H, W]
        nimages = self.preprocess_image([x["nimage"].to(self.device) for x in batched_inputs]).tensor.to(
            torch.float32)  # [B, 3, H, W]

        outputs = self.model(pimages, nimages) # ['loss', 'losses']
        # loss = outputs['loss']
        # print(loss)
        # assert 1 == 0
        # if vis:
        #     assert 1 == 0
        # print(outputs)
        # return loss

        losses = outputs['losses'] #

        return losses # dict of all kind of losses


    @torch.no_grad()
    def inference(
            self,
            batched_inputs: Tuple[Dict[str, torch.Tensor]],
            do_postprocess: bool = True,
    ):

        print("do inference...")
        # assert 1 == 0
        assert not torch.jit.is_scripting(), "Scripting for training mode is not supported."
        self.model.eval()
        print(batched_inputs[0].keys())

        pimages = self.preprocess_image([x["pimage"].to(self.device) for x in batched_inputs]).tensor.to(
            torch.float32)  # [B, 3, H, W]
        nimages = self.preprocess_image([x["nimage"].to(self.device) for x in batched_inputs]).tensor.to(
            torch.float32)  # [B, 3, H, W]

        outputs = self.model(pimages, nimages)

        return outputs

