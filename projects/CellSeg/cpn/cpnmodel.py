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
from .model import cpn2

@META_ARCH_REGISTRY.register()
class CPNModel(torch.nn.Module):

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

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
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

        images = self.preprocess_image(batched_inputs).tensor.to(torch.float32) # [B, 3, H, W]

        if self.use_freq_guild is not None:
            freq_guild = torch.cat([torch.tensor(e['low_freq_guild']).unsqueeze(0).unsqueeze(0).to(self.device) for e in batched_inputs], dim=0).to(torch.float32)  # [B, 1, H, W]
            # print(freq_guild.shape) # [B, 1, H, W]
            # assert 1 == 0
            if 'bmask' in batched_inputs[0].keys():
                bmask = torch.cat(
                    [torch.tensor(e['bmask']).unsqueeze(0).unsqueeze(0).to(self.device) for e in batched_inputs],
                    dim=0).to(torch.float32)
            else:
                bmask = None

            if self.use_freq_guild == 'concate':
                images = torch.cat([images, freq_guild], dim=1)  # [B, 4, H, W]

        else:
            freq_guild = None
            bmask = None

        # print(torch.unique(images))
        # assert 1 == 0
        # target = {}
        # target['labels'] = torch.cat([e['labels'].unsqueeze(0).to(self.device) for e in batched_inputs], dim=0).to(torch.float32)
        # target['fouriers'] = torch.cat([e['fouriers'].unsqueeze(0).to(self.device) for e in batched_inputs], dim=0).to(torch.float32)
        # target['locations'] = torch.cat([e['locations'].unsqueeze(0).to(self.device) for e in batched_inputs], dim=0).to(torch.float32)
        # target['sampled_contours'] = torch.cat([e['sampled_contours'].unsqueeze(0).to(self.device) for e in batched_inputs], dim=0).to(torch.float32)
        # target['sampling'] = torch.cat([e['sampling'].unsqueeze(0).to(self.device) for e in batched_inputs], dim=0).to(torch.float32)
        cbatched = []
        for e in batched_inputs:
            e.pop('image')
            e.pop('height')
            e.pop('width')
            e.pop('file_name')
            if self.use_freq_guild is not None:
                e.pop('low_freq_guild')
                e.pop('bmask')
            # print(e.keys())
            # assert 1 == 0
            cbatched.append(e)

        target = universal_dict_collate_fn(cbatched, device=self.device)
        if self.use_freq_guild == 'seg':
            target['low_freq_guild'] = freq_guild
            target['bmask'] = bmask

        print(images.dtype, images.shape)
        for k in target.keys():
            print(k, target[k].dtype, target[k].shape)
        # assert 1 == 0

        if self.model_seg is not None:
            output0, bseg = self.model_seg(images, target)
            assert bseg.shape[-2] == images.shape[-2] and bseg.shape[-1] == images.shape[-1]

            timages = images * (1-bseg.detach())
            output1, bseg1 = self.model_seg(timages, target)

            images = images * bseg

        outputs = self.model(images, target) # ['loss', 'losses']
        # loss = outputs['loss']
        # print(loss)
        # assert 1 == 0
        # if vis:
        #     assert 1 == 0
        # print(outputs)
        # return loss

        losses = outputs['losses'] #
        # print(losses.keys()) # dict_keys(['fourier', 'location', 'contour', 'score', 'refinement', 'boxes', 'iou', 'uncertainty', 'overall_loss'])
        # print(losses)
        # for k in losses.keys():
        #     print(k, losses[k])
        # assert 1 == 0
        if self.model_seg is not None:
            for k in output0['losses'].keys():
                losses['model_seg'+str(k)] = output0['losses'][k]
            for k in output1['losses'].keys():
                losses['model_seg1'+str(k)] = output1['losses'][k]

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
        copy_batched_inputs = [e for e in batched_inputs]
        images = self.preprocess_image(batched_inputs).tensor.to(torch.float32)

        if self.use_freq_guild is not None:
            freq_guild = torch.cat(
                [torch.tensor(e['low_freq_guild']).unsqueeze(0).unsqueeze(0).to(self.device) for e in batched_inputs],
                dim=0).to(torch.float32)  # [B, 1, H, W]
            if 'bmask' in batched_inputs[0].keys():
                bmask = torch.cat([torch.tensor(e['bmask']).unsqueeze(0).unsqueeze(0).to(self.device) for e in batched_inputs], dim=0).to(torch.float32)
            else:
                bmask = None

            if self.use_freq_guild == 'concate':
                images = torch.cat([images, freq_guild], dim=1) # [B, 4, H, W]

        else:
            freq_guild = None

        # print(torch.unique(images))
        # assert 1 == 0
        # target = {}
        # target['labels'] = torch.cat([e['labels'].unsqueeze(0).to(self.device) for e in batched_inputs], dim=0).to(torch.float32)
        # target['fouriers'] = torch.cat([e['fouriers'].unsqueeze(0).to(self.device) for e in batched_inputs], dim=0).to(torch.float32)
        # target['locations'] = torch.cat([e['locations'].unsqueeze(0).to(self.device) for e in batched_inputs], dim=0).to(torch.float32)
        # target['sampled_contours'] = torch.cat([e['sampled_contours'].unsqueeze(0).to(self.device) for e in batched_inputs], dim=0).to(torch.float32)
        # target['sampling'] = torch.cat([e['sampling'].unsqueeze(0).to(self.device) for e in batched_inputs], dim=0).to(torch.float32)

        # with torch.no_grad():
        #     outputs = self.model(images)
        cbatched = []

        for e in batched_inputs:
            te = {}
            for k in e.keys():
                if k in ('image', 'height', 'width', 'file_name', 'low_freq_guild', 'bmask'):
                    continue
                # if self.use_freq_guild!='seg' and k == 'low_freq_guild':
                #     continue

                te[k] = e[k]
            # e.pop('image')
            # e.pop('height')
            # e.pop('width')
            # e.pop('file_name')
            # print(e.keys())
            # assert 1 == 0
            cbatched.append(te)
        target = universal_dict_collate_fn(cbatched, device=self.device)
        if self.use_freq_guild == 'seg':
            target['low_freq_guild'] = freq_guild
            target['bmask'] = bmask

        print(images.dtype, images.shape)
        for k in target.keys():
            print(k, target[k].dtype if target[k] is not None else "", target[k].shape if target[k] is not None else "")
        # assert 1 == 0

        if self.model_seg is not None:
            output0, bseg = self.model_seg(images)
            assert bseg.shape[-2] == images.shape[-2] and bseg.shape[-1] == images.shape[-1]

            timages = images * (1-bseg.detach())
            output1, bseg1 = self.model_seg(timages)

            images = images * bseg

        else:
            output0 = None
            output1 = None

        outputs = self.model(images)
        # print(outputs.keys())
        # assert 1 == 0
        if 'guild_seg' in outputs:
            guild_segs = outputs.pop('guild_seg')
        else:
            guild_segs = None

        if output0 is not None:
            assert 'guild_seg' in output0
            guild_segs0 = output0.pop('guild_seg')
        else:
            guild_segs0 = None

        if output1 is not None:
            assert 'guild_seg' in output1
            guild_segs1 = output1.pop('guild_seg')
        else:
            guild_segs1 = None

            # print(type(guild_segs))
        o = util.asnumpy(outputs)
        # print(guild_segs.shape)
        # assert 1 == 0
        num = len(o['contours']) # [B, N, 224, 2]
        plt.figure(None, (13 * num, 13))

        print(len(copy_batched_inputs))
        print(copy_batched_inputs[0].keys())
        compute_metric = self.compute_metric
        for idx in range(num):
            # rname = time.time()
            print(copy_batched_inputs[idx]['file_name'])
            rname = copy_batched_inputs[idx]['file_name'] + str(time.time()) if not self.training else str(time.time())

            image = util.asnumpy(copy_batched_inputs[idx]['image']) # [3, H, W]
            # plt.subplot(2, num, idx + 1)
            # print(o['scores'][idx])
            mask = o['scores'][idx] >= 0.9
            if guild_segs is not None:
                guild_seg = guild_segs[idx]
            else:
                guild_seg = None

            if compute_metric and not self.training:
                contours = o['contours'][idx][mask]
                # print(len(contours))
                # print(type(contours)) # np.array
                # print(contours.shape)
                # print(contours[0].shape)
                # contour_edge = contour_to_pixel_edge(torch.tensor(contours[4]), torch.tensor([512, 512]))
                # print(contour_edge.shape)
                # visual_pixel_edges(contour_edge)
                #
                # tmp = np.zeros((CellDatasetMapper.unmap(image.copy().transpose(1, 2, 0)).shape[:2]), dtype=np.int32)
                # cv2.drawContours(tmp, [np.expand_dims(contours[4], axis=1).astype(np.int32)], -1, 1, cv2.FILLED)
                # plt.imshow(tmp)
                # plt.show()
                #
                # assert 1 == 0
                # for i in range(len(contours)):
                #     print(contours[i].shape, type(contours[i])) # [224, 2]
                # print(contours)
                pred_ins_img = np.zeros((CellDatasetMapper.unmap(image.copy().transpose(1, 2, 0)).shape[:2]), dtype=np.int32)
                print(pred_ins_img.shape)
                ccontours = [np.expand_dims(contours[i], axis=1).astype(np.int32) for i in range(len(contours))]
                print(len(ccontours))
                for j, e in enumerate(ccontours):
                    if guild_seg is not None:
                        tmp = np.zeros((CellDatasetMapper.unmap(image.copy().transpose(1, 2, 0)).shape[:2]), dtype=np.int32)
                        cv2.drawContours(tmp, [e], -1, 1, cv2.FILLED)
                        # check
                        tmp = tmp * guild_seg
                        if np.max(tmp) < 0.0:

                            print("##############################")
                            print('filter instance ', j)
                            continue

                    # print(e.shape) # [224, 1, 2]
                    cv2.drawContours(pred_ins_img, [e], -1, (j+1), cv2.FILLED)
                # cv2.imwrite("./ins_seg.png", white_img)
                # plt.subplot(141)
                # plt.imshow(CellDatasetMapper.unmap(image.copy().transpose(1, 2, 0)))
                # plt.subplot(142)
                # plt.imshow(pred_ins_img)
                white_img = np.zeros((CellDatasetMapper.unmap(image.copy().transpose(1, 2, 0)).shape[:2]))
                cv2.drawContours(white_img, ccontours, -1, 255, cv2.FILLED)
                # plt.subplot(143)
                # plt.imshow(white_img)

                tcontours = copy_batched_inputs[idx]['sampled_contours'] #
                # print(type(tcontours)) # tuple
                # print(len(tcontours)) # 1
                # print(type(tcontours[0])) # array
                # print(tcontours[0].shape) # [n_ins, 224, 2]
                true_ins_img = np.zeros((CellDatasetMapper.unmap(image.copy().transpose(1, 2, 0)).shape[:2]), dtype=np.int32)
                ccontours = [np.expand_dims(tcontours[0][i], axis=1).astype(np.int32) for i in range(len(tcontours[0]))]
                print(len(ccontours))
                for j, e in enumerate(ccontours):
                    # print(e.shape) # [224, 1, 2]
                    cv2.drawContours(true_ins_img, [e], -1, (j+1), cv2.FILLED)
                # plt.subplot(144)
                # plt.imshow(true_ins_img)
                # plt.show()


                # maskfiles = os.path.join(
                #     "/run/user/1000/gvfs/smb-share:server=192.168.2.221,share=fqpathotech/ lingpeng/MoNuSeg 2018 Training Data/MoNuSeg 2018 Training Data/mask_512",
                #     os.path.splitext(copy_batched_inputs[idx]['file_name'])[0]
                # )
                # # print(maskfiles)
                # true_ins_img = np.zeros((CellDatasetMapper.unmap(image.copy().transpose(1, 2, 0)).shape[:2]),
                #                         dtype=np.int32)
                # for j in range(len(os.listdir(maskfiles))):
                #     # print(os.path.join(
                #     #     maskfiles, os.listdir(maskfiles)[j]
                #     # ))
                #     mj = cv2.imread(os.path.join(
                #         maskfiles, os.listdir(maskfiles)[j]
                #     ), 3)
                #     mj = cv2.cvtColor(mj, cv2.COLOR_BGR2GRAY)
                #     # plt.subplot(121)
                #     # plt.imshow(mj)
                #     # plt.show()
                #     # print(mj.shape)
                #     emask = (mj != 0) & (true_ins_img == 0)
                #     true_ins_img[emask] = j+1
                #     # plt.subplot(122)
                #     # plt.imshow(true_ins_img)
                #     # plt.show()
                #     # assert 1 == 0




                ajis1 = compute_aji(remap_label(np.copy(true_ins_img)), remap_label(np.copy(pred_ins_img)))
                self.ajis.append(ajis1)

                res = get_fast_pq(remap_label(np.copy(true_ins_img)), remap_label(np.copy(pred_ins_img)))
                dq, sq, pq = res[0]
                self.dq.append(dq)
                self.sq.append(sq)
                self.pq.append(pq)

                continue

            contours = o['contours'][idx][mask]

            if guild_seg is not None:
                ccontours = [np.expand_dims(contours[i], axis=1).astype(np.int32) for i in range(len(contours))]
                tcontours = []
                for j, e in enumerate(ccontours):
                    tmp = np.zeros((CellDatasetMapper.unmap(image.copy().transpose(1, 2, 0)).shape[:2]), dtype=np.int32)
                    cv2.drawContours(tmp, [e], -1, 1, cv2.FILLED)
                    # check
                    tmp = tmp * guild_seg
                    if np.max(tmp) < 0.0:
                        print("##############################")
                        print('filter instance ', j)
                        continue

                    tcontours.append(e)
                ccontours = np.array([e.squeeze(1) for e in tcontours]) # [n_ins, 224, 2]
            else:
                ccontours = contours
            # print(guild_seg.shape)
            # plt.subplot(1, 2, 1)
            # plt.imshow(CellDatasetMapper.unmap(image.copy().transpose(1, 2, 0)))
            # plt.subplot(1, 2, 2)
            # plt.imshow(guild_seg)
            # plt.show()
            # assert 1 == 0
            vis.show_detection(
                CellDatasetMapper.unmap(image.copy().transpose(1, 2, 0)),
                contours=ccontours,
                contour_line_width=3,
                # scores=o['scores'][idx][mask],
            )
            # plt.show()
            plt.savefig(os.path.join(self.vis_file, str(rname)+'_pred.png'))
            plt.close()

            plt.figure(None, (13 * num, 13))
            # plt.subplot(2, num, idx + 1)
            if 'sampled_contours' in copy_batched_inputs[idx].keys():
                tcontours = torch.tensor(copy_batched_inputs[idx]['sampled_contours']).squeeze(0)
                print(tcontours.shape)
                # assert 1 == 0
                vis.show_detection(
                    CellDatasetMapper.unmap(image.copy().transpose(1, 2, 0)),
                    contours=tcontours,
                    contour_line_width=3,
                )
                # plt.show()
                plt.savefig(os.path.join(self.vis_file, str(rname)+'_gt.png'))
                plt.close()

            if guild_segs is not None:
                guild_seg = guild_segs[idx]
                plt.subplot(1, 3 if guild_segs0 is None else 5, 1)
                plt.imshow(guild_seg)
                plt.subplot(1, 3 if guild_segs0 is None else 5, 2)
                plt.imshow(copy_batched_inputs[idx]['low_freq_guild'])
                if 'bmask' in copy_batched_inputs[idx].keys():
                    plt.subplot(1, 3 if guild_segs0 is None else 5, 3)
                    plt.imshow(np.clip(copy_batched_inputs[idx]['low_freq_guild']+copy_batched_inputs[idx]['bmask'], a_min=0.0, a_max=1.0))

                if guild_segs0 is not None:
                    plt.subplot(1, 5, 4)
                    plt.imshow(guild_segs0[idx])
                if guild_segs1 is not None:
                    plt.subplot(1, 5, 5)
                    plt.imshow(guild_segs1[idx])

                # plt.show()
                plt.savefig(os.path.join(self.vis_file, str(rname)+'_seg.png'))
                plt.close()
            # assert 1 == 0
            if self.training:
                break

        # plt.close()
        print("AJI: ", np.mean(self.ajis))
        print("DQ: ", np.mean(self.dq))
        print("SQ: ", np.mean(self.sq))
        print("PQ: ", np.mean(self.pq))
        return outputs


def contour_to_pixel_edge(contours, image_size):
    # print(contours)
    nsize = 256
    # contours_coords = (contours / image_size)*2 - 1 # normalize coord
    contours = contours * (nsize/512.)
    pixel_edge = torch.zeros((nsize, nsize), requires_grad=True)
    grid_indices = contours.round()
    pixel_edge[grid_indices[:, 1].long(), grid_indices[:, 0].long()] = 1
    grid_indices = contours.ceil()
    pixel_edge[grid_indices[:, 1].long(), grid_indices[:, 0].long()] = 1
    grid_indices = contours.floor()
    pixel_edge[grid_indices[:, 1].long(), grid_indices[:, 0].long()] = 1
    grid_indices = contours.trunc()
    pixel_edge[grid_indices[:, 1].long(), grid_indices[:, 0].long()] = 1
    pixel_edge[pixel_edge > 0] = 1.
    print(pixel_edge.requires_grad)
    return pixel_edge

def visual_pixel_edges(pixel_edges):
    plt.imshow(pixel_edges)
    # plt.colorbar()
    plt.show()
    plt.close()


def compute_aji(true, pred): # aggregated jaccard index
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
        # print(true_id, len(true_masks))
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

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-16)
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

def get_fast_pq(true, pred, match_iou=0.5):
    assert match_iou >= 0.0
    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    if len(pred_id_list) == 1:
        return [0, 0, 0], [0, 0, 0, 0]

    true_masks = [None, ]
    for t in true_id_list[1: ]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None, ]
    for p in pred_id_list[1: ]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    pairwise_iou = np.zeros(
        [len(true_id_list)-1, len(pred_id_list)-1], dtype=np.float64
    )

    for true_id in true_id_list[1: ]:
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0: # background
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id-1, pred_id-1] = iou

    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1
        paired_pred += 1
    else:
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]

        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]


    unpaired_true = [idx for idx in true_id_list[1: ] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1: ] if idx not in paired_pred]

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)

    dq = tp / (tp + 0.5*fp + 0.5*fn)
    sq = paired_iou.sum() / (tp + 1.0e-16)
    pq = dq * sq
    # [dq, sq, pq
    return [dq, sq, pq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


def remap_label(pred, by_size=False):
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred

    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred