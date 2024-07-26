import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List, Union
import warnings
from pytorch_lightning.core.mixins import HyperparametersMixin
import math

from util.util import add_to_loss_dict, reduce_loss_dict
from .commons import ScaledTanh, ReadOut, Fuse2d
from backbone.unet import ResNeXt101UNet, ResNet101UNet, ResNet18UNet, ResNet50UNet, ResNeXt50UNet, U22
from backbone.fpn import ResNeXt101FPN, ResNet101FPN
from ops.cpn import rel_location2abs_location, fouriers2contours, scale_contours, scale_fourier, batched_box_nmsi, order_weighting, resolve_refinement_buckets
from ops import boxes as bx
from ops.commons import downsample_labels
from .loss import IoULoss, BoxNpllLoss

__all__ = []

def dummy_loss(*a, sub=1):
    return 0. * sum([i[:sub].mean() for i in a if isinstance(i, Tensor)])

def values2bins(values, limits, bins):
    mi, ma = limits
    v = (values - mi) / (ma - mi)
    ma -= mi
    return ((v // (ma / bins)) % bins).int()

def register(obj):
    __all__.append(obj.__name__)
    return obj


def resolve_batch_index(inputs: dict, n, b) -> dict:
    outputs = OrderedDict({k: (None if v is None else []) for k, v in inputs.items()})
    for batch_index in range(n):
        sel = b == batch_index
        for k, v in inputs.items():
            o = outputs[k]

            if o is not None:
                o.append(v[sel])
    return outputs


def resolve_keep_indices(inputs: dict, keep: list) -> dict:
    outputs = OrderedDict({k: (None if v is None else []) for k, v in inputs.items()})
    for j, indices in enumerate(keep):
        for k, v in inputs.items():
            o = outputs[k]
            if o is not None:
                o.append(v[j][indices])
    return outputs


def local_refinement(det_indices, refinement, num_loops, num_buckets, original_size, sampling, b):
    all_det_indices = []
    for _ in torch.arange(0, num_loops):
        det_indices = torch.round(det_indices.detach())  # Tensor[num_contours, samples, 2]
        det_indices[..., 0].clamp_(0, original_size[1] - 1)
        det_indices[..., 1].clamp_(0, original_size[0] - 1)
        indices = det_indices.detach().long()  # Tensor[-1, samples, 2]
        if num_buckets == 1:
            responses = refinement[b[:, None], :, indices[:, :, 1], indices[:, :, 0]]  # Tensor[-1, samples, 2]
        else:
            buckets = resolve_refinement_buckets(sampling, num_buckets)
            responses = None
            for bucket_indices, bucket_weights in buckets:
                bckt_idx = torch.stack((bucket_indices * 2, bucket_indices * 2 + 1), -1)
                cur_ref = refinement[b[:, None, None], bckt_idx, indices[:, :, 1, None], indices[:, :, 0, None]]
                cur_ref = cur_ref * bucket_weights[..., None]
                if responses is None:
                    responses = cur_ref
                else:
                    responses = responses + cur_ref
        det_indices = det_indices + responses
        all_det_indices.append(det_indices)
    return det_indices, all_det_indices


def _resolve_channels(encoder_channels, backbone_channels, keys: Union[list, tuple, str], encoder_prefix: str):
    channels = 0
    reference = None
    if not isinstance(keys, (list, tuple)):
        keys = [keys]
    for k in keys:
        if k.startswith(encoder_prefix):
            channels += encoder_channels[int(k[len(encoder_prefix):])]
        else:
            channels += backbone_channels[int(k)]
        if reference is None:
            reference = channels
    return channels, reference, len(keys)


def _resolve_features(features, keys):
    if isinstance(keys, (tuple, list)):
        return [features[k] for k in keys]
    return features[keys]

def _resolve_features_map(features, keys):
    if isinstance(keys, (tuple, list)):
        # return [features[k] for k in keys]
        return {k : features[k] for k in keys}
    # return features[keys]
    return {keys: features[keys]}

def _equal_size(x, reference, mode='bilinear', align_corners=False):
    if reference.shape[2:] != x.shape[2:]:  # 337 ns
        # bilinear: 3.79 ms for (128, 128) to (512, 512)
        # bicubic: 11.5 ms for (128, 128) to (512, 512)
        x = F.interpolate(x, reference.shape[2:],
                          mode=mode, align_corners=align_corners)

    return x


def _apply_score_bounds(scores, scores_lower_bound, scores_upper_bound):
    if scores_upper_bound is not None:
        scores = torch.minimum(scores, _equal_size(scores_upper_bound, scores))
    if scores_lower_bound is not None:
        scores = torch.maximum(scores, _equal_size(scores_lower_bound, scores))
    return scores


class CPNCore(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            backbone_channels,
            order,
            score_channels: int,
            refinement: bool = True,
            refinement_margin: float = 3.,
            uncertainty_head=False,
            contour_features='1',
            location_features='1',
            uncertainty_features='1',
            score_features='1',
            refinement_features='0',
            contour_head_channels=None,
            contour_head_stride=1,
            refinement_head_channels=None,
            refinement_head_stride=1,
            refinement_interpolation='bilinear',
            refinement_buckets=1,
            refinement_full_res=True,
            encoder_channels=None,
            **kwargs,
    ):
        super().__init__()
        self.order = order
        self.backbone = backbone
        self.refinement_interpolation = refinement_interpolation
        assert refinement_buckets >= 1
        self.refinement_buckets = refinement_buckets

        if encoder_channels is None:
            encoder_channels = backbone_channels  # assuming same channels
        channels = encoder_channels, backbone_channels # ([64, 256, 512, 1024, 2048, 256], [64, 256, 512, 1024, 2048, 256])
        # print(channels)
        # assert 1 == 0
        kw = {'encoder_prefix': kwargs.get('encoder_prefix', 'encoder.')}
        self.contour_features = contour_features
        self.location_features = location_features
        self.score_features = score_features
        self.refinement_features = refinement_features
        self.uncertainty_features = uncertainty_features
        self.refinement_full_res = refinement_full_res
        fourier_channels, fourier_channels_, num_fourier_inputs = _resolve_channels(*channels, contour_features, **kw)
        loc_channels, loc_channels_, num_loc_inputs = _resolve_channels(*channels, location_features, **kw)
        sco_channels, sco_channels_, num_score_inputs = _resolve_channels(*channels, score_features, **kw)
        ref_channels, ref_channels_, num_ref_inputs = _resolve_channels(*channels, refinement_features, **kw)
        unc_channels, unc_channels_, num_unc_inputs = _resolve_channels(*channels, uncertainty_features, **kw)


        fuse_kw = kwargs.get('fuse_kwargs', {})

        self.use_guild = kwargs.get("use_guild", None)
        # print(self.use_guild)
        # assert 1 == 0

        dropout = 0.1
        # Score
        self.score_fuse = Fuse2d(sco_channels, sco_channels_, **fuse_kw) if num_score_inputs > 1 else None
        self.score_head = ReadOut(
            sco_channels_, score_channels,
            kernel_size=kwargs.get('kernel_size_score', 7),
            padding=kwargs.get('kernel_size_score', 7) // 2,
            channels_mid=contour_head_channels,
            stride=contour_head_stride,
            dropout=dropout,
            activation=kwargs.pop('head_activation_score', kwargs.get('head_activation', 'relu'))
        )

        # Location
        self.location_fuse = Fuse2d(loc_channels, loc_channels_, **fuse_kw) if num_loc_inputs > 1 else None
        self.location_head = ReadOut(
            loc_channels_, 2,
            kernel_size=kwargs.get('kernel_size_location', 7),
            padding=kwargs.get('kernel_size_location', 7) // 2,
            channels_mid=contour_head_channels,
            stride=contour_head_stride,
            dropout=dropout,
            activation=kwargs.pop('head_activation_location', kwargs.get('head_activation', 'relu'))
        )

        # Fourier
        self.fourier_fuse = Fuse2d(fourier_channels, fourier_channels_, **fuse_kw) if num_fourier_inputs > 1 else None
        self.fourier_head = ReadOut(
            fourier_channels_, order * 4,
            kernel_size=kwargs.get('kernel_size_fourier', 7),
            padding=kwargs.get('kernel_size_fourier', 7) // 2,
            channels_mid=contour_head_channels,
            stride=contour_head_stride,
            dropout=dropout,
            activation=kwargs.pop('head_activation_fourier', kwargs.get('head_activation', 'relu'))
        )

        # Uncertainty
        if uncertainty_head:
            self.uncertainty_fuse = Fuse2d(unc_channels, unc_channels_, **fuse_kw) if num_unc_inputs > 1 else None
            self.uncertainty_head = ReadOut(
                unc_channels_, 4,
                kernel_size=kwargs.get('kernel_size_uncertainty', 7),
                padding=kwargs.get('kernel_size_uncertainty', 7) // 2,
                channels_mid=contour_head_channels,
                stride=contour_head_stride,
                final_activation='sigmoid',
                dropout=dropout,
                activation=kwargs.pop('head_activation_uncertainty', kwargs.get('head_activation', 'relu'))
            )
        else:
            self.uncertainty_fuse = self.uncertainty_head = None

        # Refinement
        if refinement:
            self.refinement_fuse = Fuse2d(ref_channels, ref_channels_, **fuse_kw) if num_ref_inputs > 1 else None
            self.refinement_head = ReadOut(
                ref_channels_, 2 * refinement_buckets,
                kernel_size=kwargs.get('kernel_size_refinement', 7),
                padding=kwargs.get('kernel_size_refinement', 7) // 2,
                final_activation=ScaledTanh(refinement_margin),
                channels_mid=refinement_head_channels,
                stride=refinement_head_stride,
                dropout=dropout,
                activation=kwargs.pop('head_activation_refinement', kwargs.get('head_activation', 'relu'))
            )
        else:
            self.refinement_fuse = self.refinement_head = None

        self.fusion_type = kwargs.get("fusion_type", 'fusion')
        if self.use_guild == 'seg':
            self.guild_seg_fuse = None
            self.guild_seg_head = ReadOut(
                ref_channels_, 1,
                kernel_size=kwargs.get('kernel_size_uncertainty', 7),
                padding=kwargs.get('kernel_size_uncertainty', 7) // 2,
                channels_mid=contour_head_channels,
                stride=contour_head_stride,
                final_activation='sigmoid',
                dropout=dropout,
                activation=kwargs.pop('head_activation_uncertainty', kwargs.get('head_activation', 'relu'))
            )
            self.guild_features = kwargs.get("guild_features", '0')

            self.feature_fusion = nn.ModuleDict()
            self.tfeature_fusion = nn.ModuleDict()

            for i in range(len(encoder_channels)):
                if str(i) in self.guild_features:
                    # if len(self.guild_features) > 1:
                    self.feature_fusion[str(i)] = nn.Conv2d(encoder_channels[i], ref_channels_, 1, bias=False)

                if self.fusion_type == 'fusion':
                    self.tfeature_fusion[str(i)] = nn.Sequential(
                        nn.Conv2d(encoder_channels[i] * 2, encoder_channels[i], 1, bias=False)

                    )

                elif self.fusion_type == 'attn':

                    self.tfeature_fusion[str(i)] = nn.Sequential(
                        nn.Conv2d(encoder_channels[i], encoder_channels[i]//16, 1, bias=False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(encoder_channels[i]//16, encoder_channels[i], 1, bias=False),
                    )
                    # spatial attention
                    self.cfeature_fusion = nn.Sequential(
                        nn.Conv2d(2, 1, 3, padding=1, bias=False),
                        nn.Sigmoid(),

                    )



                elif self.fusion_type == 'dattn':
                    # print(len(self.guild_features), self.guild_features)
                    # assert 1 == 0
                    pass
                    # if str(i) in self.guild_features:
                    # # if len(self.guild_features) > 1:
                    #     self.feature_fusion[str(i)] = nn.Sequential(
                    #         nn.Conv2d(encoder_channels[i], ref_channels_, 1, bias=False),
                    #     )

        else:
            self.guild_seg_fuse = self.guild_seg_head = None
        # print(self.use_guild, self.guild_seg_fus, self.guild_seg_head)
        # assert 1 == 0

    def forward(self, inputs):
        features, guild_features = self.backbone(inputs)
        # features = self.backbone(inputs)
        # guild_features = None

        # guild seg
        # print(guild_features.shape)
        if self.use_guild is not None and self.use_guild=='seg':
            if guild_features is None and self.fusion_type=='dattn':
                if not isinstance(features, torch.Tensor):
                    guild_features = _resolve_features(features, self.guild_features) # list

                else:
                    guild_features = features


        if self.guild_seg_head is not None:
            if self.guild_seg_fuse is not None:
                # assert 1 == 0
                guild_features = self.guild_seg_fuse(guild_features)
            # full size
            # print(guild_features.shape)
            if isinstance(guild_features, list) or isinstance(guild_features, dict):
                # assert 1 == 0
                guild_seg = {}
                for idx, k in enumerate(self.guild_features):
                    guild_seg[k] = self.guild_seg_head(self.feature_fusion[k](guild_features[k])) # [B, 1, h, w]

            else:
                guild_seg = _equal_size(self.guild_seg_head(guild_features), inputs) # [B, 1, H, W] after sigmoid

        else:
            guild_seg = None


        if guild_features is not None and self.fusion_type != 'dattn':
            # assert 1 == 0
            cfeatures = {}
            for k in features.keys():
                if k not in self.feature_fusion.keys():
                    cfeatures[k] = features[k]
                    # continue
                else:
                    # print(k)
                    B, C, h, w = features[k].shape
                    featuresi = features[k]   # [B, C, h, w]
                    # print(k, guild_features.keys())
                    guild_featuresi = guild_features[k]   # [B, C, h, w]
                    # print(featuresi.shape, guild_featuresi.shape)
                    if self.fusion_type == 'attn':

                        guild_featuresi_max = self.tfeature_fusion[k](F.adaptive_max_pool2d(guild_featuresi, (1,1))) # [B, C, 1, 1]
                        guild_featuresi_mean = self.tfeature_fusion[k](F.adaptive_avg_pool2d(guild_featuresi, (1,1))) # [B, C, 1, 1]
                        guild_featuresi = F.sigmoid(guild_featuresi_max+guild_featuresi_mean) * guild_featuresi # [B, C, h, w]

                        guild_featuresi_max = torch.max(guild_featuresi, dim=1, keepdim=True)[0]
                        guild_featuresi_mean = torch.mean(guild_featuresi, dim=1, keepdim=True)
                        guild_featuresi = torch.cat([guild_featuresi_mean, guild_featuresi_max], dim=1) # [B, 2, h, w]
                        cfeatures[k] = self.cfeature_fusion(guild_featuresi) * featuresi


                    else:
                        cfeatures[k] = self.tfeature_fusion[k](torch.cat([featuresi, guild_featuresi], dim=1))
                    # print(cfeatures[k].shape)

            # print(features.keys()) # dict_keys(['out', '0', '1', '2', '3', '4', '5', 'encoder.0', 'encoder.1', 'encoder.2', 'encoder.3', 'encoder.4', 'encoder.5'])
            # print(guild_features.keys()) # dict_keys(['out', '0', '1', '2', '3', '4', '5'])
            # assert 1 == 0
            features = cfeatures

        if guild_features is not None and self.fusion_type == 'dattn':
            cfeatures = {}
            for k in features.keys():
                B, C, h, w = features[k].shape
                featuresi = features[k]  # [B, C, h, w]
                if isinstance(guild_seg, dict):
                    if k in guild_seg.keys():
                        guild_featuresi = guild_seg[k]
                        assert guild_featuresi.shape[-2: ] == featuresi.shape[-2: ]
                    else:
                        assert '0' in guild_seg.keys()
                        guild_featuresi = _equal_size(guild_seg['0'], featuresi)

                else:
                    guild_featuresi = _equal_size(guild_seg, featuresi) # [B, 1, h, w] same as featuresi

                featuresi = featuresi * guild_featuresi + featuresi
                cfeatures[k] = featuresi

            features = cfeatures

        # print(type(features)) # <class 'collections.OrderedDict'>
        # if isinstance(features, dict):
        #     for k in features.keys():
        #         print(k, features[k].shape)
        #     if self.use_guild is not None:
        #         print("guild features...") # same as x
        #         for k in guild_features.keys():
        #             print(k, guild_features[k].shape)
        # assert 1 == 0

        if isinstance(features, torch.Tensor):
            score_features = fourier_features = location_features = unc_features = ref_features = features
            # assert 1 == 0
        else:
            score_features = _resolve_features(features, self.score_features)
            fourier_features = _resolve_features(features, self.contour_features)
            location_features = _resolve_features(features, self.location_features)
            unc_features = _resolve_features(features, self.uncertainty_features)
            ref_features = _resolve_features(features, self.refinement_features)
            # print(type(score_features))
            # assert 1 == 0
        # print(guild_features.shape)
        # assert 1 == 0
        # Scores
        if self.score_fuse is not None:
            score_features = self.score_fuse(score_features)
        scores = self.score_head(score_features)

        # Locations
        if self.location_fuse is not None:
            location_features = self.location_fuse(location_features)
        locations = self.location_head(location_features)

        # Fourier
        if self.fourier_fuse is not None:
            fourier_features = self.fourier_fuse(fourier_features)
        fourier = self.fourier_head(fourier_features)

        # Uncertainty
        if self.uncertainty_head is not None:
            if self.uncertainty_fuse is not None:
                unc_features = self.uncertainty_fuse(unc_features)
            uncertainty = self.uncertainty_head(unc_features)
        else:
            uncertainty = None

        # Refinement
        if self.refinement_head is not None:
            if self.refinement_fuse is not None:
                ref_features = self.refinement_fuse(ref_features)
            if self.refinement_full_res:
                ref_features = _equal_size(ref_features, inputs, mode=self.refinement_interpolation)
            refinement = _equal_size(self.refinement_head(ref_features), inputs, mode=self.refinement_interpolation)
        else:
            refinement = None


        # print(guild_seg.shape)  # [B, 1, H, W]
        # assert 1 == 0
        return scores, locations, refinement, fourier, uncertainty, guild_seg, features



class CPN(nn.Module, HyperparametersMixin):
    def __init__(
            self,
            backbone: nn.Module,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            certainty_thresh: float = None,
            samples: int = 32,
            classes: int = 2,

            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,

            contour_features='1',
            location_features='1',
            uncertainty_features='1',
            score_features='1',
            refinement_features='0',

            uncertainty_head=False,
            uncertainty_nms=False,
            uncertainty_factor=7.,

            contour_head_channels=None,
            contour_head_stride=1,
            order_weights=True,
            refinement_head_channels=None,
            refinement_head_stride=1,
            refinement_interpolation='bilinear',

            **kwargs
    ):
        """CPN base class.

        This is the base class for the Contour Proposal Network.

        References:
            https://www.sciencedirect.com/science/article/pii/S136184152200024X

        Args:
            backbone: A backbone network. E.g. ``cd.models.U22(in_channels, 0)``.
            order: Contour order. The higher, the more complex contours can be proposed.
                ``order=1`` restricts the CPN to propose ellipses, ``order=3`` allows for non-convex rough outlines,
                ``order=8`` allows even finer detail.
            nms_thresh: IoU threshold for non-maximum suppression (NMS). NMS considers all objects with
                ``iou > nms_thresh`` to be identical.
            score_thresh: Score threshold. For binary classification problems (object vs. background) an object must
                have ``score > score_thresh`` to be proposed as a result.
            samples: Number of samples. This sets the number of coordinates with which a contour is defined.
                This setting can be changed on the fly, e.g. small for training and large for inference.
                Small settings reduces computational costs, while larger settings capture more detail.
            classes: Number of classes. Default: 2 (object vs. background).
            refinement: Whether to use local refinement or not. Local refinement generally improves pixel precision of
                the proposed contours.
            refinement_iterations: Number of refinement iterations.
            refinement_margin: Maximum refinement margin (step size) per iteration.
            refinement_buckets: Number of refinement buckets. Bucketed refinement is especially recommended for data
                with overlapping objects. ``refinement_buckets=1`` practically disables bucketing,
                ``refinement_buckets=6`` uses 6 different buckets, each influencing different fractions of a contour.
            contour_features: If ``backbone`` returns a dictionary of features, this is the key used to retrieve
                the features that are used to predict contours.
            refinement_features: If ``backbone`` returns a dictionary of features, this is the key used to retrieve
                the features that are used to predict the refinement tensor.
            contour_head_channels: Number of intermediate channels in contour ``ReadOut`` Modules. By default, this is
                the number of incoming feature channels.
            contour_head_stride: Stride used for the contour prediction. Larger stride means less contours can
                be proposed in total, which speeds up execution times.
            order_weights: Whether to use order specific weights.
            refinement_head_channels: Number of intermediate channels in refinement ``ReadOut`` Modules. By default,
                this is the number of incoming feature channels.
            refinement_head_stride: Stride used for the refinement prediction. Larger stride means less detail, but
                speeds up execution times.
            refinement_interpolation: Interpolation mode that is used to ensure that refinement tensor and input
                image have the same shape.
            score_encoder_features: Whether to use encoder-head skip connections for the score head.
            refinement_encoder_features: Whether to use encoder-head skip connections for the refinement head.
        """
        super().__init__()

        self.num_classes = classes
        self.order = order
        self.nms_thresh = nms_thresh
        self.samples = samples
        self.score_thresh = score_thresh
        self.score_channels = 1 if classes in (1, 2) else classes
        self.refinement = refinement
        self.refinement_iterations = refinement_iterations
        self.refinement_margin = refinement_margin
        self.functional = False
        self.full_detail = False
        self.score_target_dtype = None
        self.certainty_thresh = certainty_thresh
        self.uncertainty_nms = uncertainty_nms

        if not hasattr(backbone, 'out_channels'):
            raise ValueError('Backbone should have an attribute out_channels that states the channels of its output.')

        self.core = kwargs.get('core_cls', CPNCore)(
            backbone=backbone,
            backbone_channels=backbone.out_channels,
            order=order,
            score_channels=self.score_channels,
            refinement=refinement,
            refinement_margin=refinement_margin,
            contour_features=contour_features,
            location_features=location_features,
            uncertainty_features=uncertainty_features,
            score_features=score_features,
            refinement_features=refinement_features,
            contour_head_channels=contour_head_channels,
            contour_head_stride=contour_head_stride,
            refinement_head_channels=refinement_head_channels,
            refinement_head_stride=refinement_head_stride,
            refinement_interpolation=refinement_interpolation,
            refinement_buckets=refinement_buckets,
            uncertainty_head=uncertainty_head,
            **kwargs
        )

        # print(order_weighting(self.order).shape)
        # assert 1 == 0
        if isinstance(order_weights, bool):
            if order_weights:
                # print("register order_weights module")
                # assert 1 == 0
                self.register_buffer('order_weights', order_weighting(self.order))
                # self.order_weights = order_weighting(self.order)
            else:
                self.order_weights = 1.
        else:
            self.order_weights = order_weights

        self.objectives = OrderedDict({
            'score': nn.CrossEntropyLoss() if self.score_channels > 1 else nn.BCEWithLogitsLoss(),
            'fourier': nn.L1Loss(reduction='none'),
            'location': nn.L1Loss(),
            'contour': nn.L1Loss(),
            'refinement': nn.L1Loss() if refinement else None,
            'boxes': None,
            'iou': IoULoss(min_size=1.),
            'uncertainty': BoxNpllLoss(uncertainty_factor, min_size=1., sigmoid=False) if uncertainty_head else None
        })
        self.weights = {
            'fourier': 1.,  # note: fourier has order specific weights
            'location': 1.,
            'contour': 3.,
            'score_bg': 1.,
            'score_fg': 1.,
            'refinement': 1.,
            'boxes': .88,
            'iou': 1.,
            'uncertainty': 1.,
        }

        self._rel_location2abs_location_cache: Dict[str, Tensor] = {}
        self._fourier2contour_cache: Dict[str, Tensor] = {}
        self._warn_iou = False

        self.ignore_bg = kwargs.get("ignore_bg", None)
        if self.ignore_bg:
            print("#"*20)
            print("ignore background during training...")

        self.use_guild = kwargs.get("use_guild", None)

        self.use_topo_loss = kwargs.get("use_topo_loss", False)
        self.use_contr = kwargs.get("use_contr", False)

    def compute_loss(
            self,
            uncertainty,
            fourier,
            locations,
            contours,
            refined_contours,
            all_refined_contours,
            boxes,
            raw_scores,
            targets: dict,
            labels,
            fg_masks,
            sampling,
            b
    ):
        assert targets is not None

        fourier_targets = targets.get('fourier')
        location_targets = targets.get('locations')
        contour_targets = targets.get('sampled_contours')
        hires_contour_targets = targets.get('hires_sampled_contours')
        box_targets = targets.get('boxes')
        class_targets = targets.get('classes')

        losses = OrderedDict({
            'fourier': None,
            'location': None,
            'contour': None,
            'score': None,
            'refinement': None,
            'boxes': None,
            'iou': None,
            'uncertainty': None,
        })

        bg_masks = labels == 0
        fg_n, fg_y, fg_x = torch.where(fg_masks)
        bg_n, bg_y, bg_x = torch.where(bg_masks)
        objectives = self.objectives

        fg_scores = raw_scores[fg_n, :, fg_y, fg_x]  # Tensor[-1, classes]
        bg_scores = raw_scores[bg_n, :, bg_y, bg_x]  # Tensor[-1, classes]
        fg_indices = labels[fg_n, fg_y, fg_x].long() - 1  # -1 because fg labels start at 1, but indices at 0
        fg_num = fg_indices.numel()
        bg_num = bg_scores.numel()

        if box_targets is not None:
            if fg_num:
                box_targets = box_targets[b, fg_indices]
        elif not self._warn_iou and self.objectives.get('iou') is not None and self.samples < 32:
            self._warn_iou = True
            warnings.warn('The iou loss option of the CPN is enabled, but the `samples` setting is rather low. '
                          'This may impair detection performance. '
                          'Increase `samples`, provide box targets manually or set model.objectives["iou"] = False.')
        if fg_num and contour_targets is not None:
            # print(contour_targets.shape)
            # assert 1 == 0
            c_tar = contour_targets[b, fg_indices]  # Tensor[num_pixels, samples, 2]

            if box_targets is None:
                box_targets = bx.contours2boxes(c_tar, axis=1)

        if self.score_target_dtype is None:
            if isinstance(objectives['score'], nn.CrossEntropyLoss):
                self.score_target_dtype = torch.int64
            else:
                self.score_target_dtype = fg_scores.dtype

        if fg_num:
            if class_targets is None:
                ones = torch.broadcast_tensors(torch.ones((), dtype=self.score_target_dtype, device=fg_scores.device),
                                               fg_scores[..., 0])[0]
            else:
                ones = class_targets[b, fg_indices].to(self.score_target_dtype)
            if self.score_channels == 1:
                fg_scores = torch.squeeze(fg_scores, 1)
            add_to_loss_dict(losses, 'score', objectives['score'](fg_scores, ones), self.weights['score_fg'])

        if bg_num:
            zeros = torch.broadcast_tensors(torch.zeros((), dtype=self.score_target_dtype, device=bg_scores.device),
                                            bg_scores[..., 0])[0]
            if self.score_channels == 1:
                bg_scores = torch.squeeze(bg_scores, 1)
            add_to_loss_dict(losses, 'score', objectives['score'](bg_scores, zeros), self.weights['score_bg'])

        if fg_num:
            if fourier_targets is not None:
                f_tar = fourier_targets[b, fg_indices]  # Tensor[num_pixels, order, 4]
                add_to_loss_dict(losses, 'fourier',
                                 (objectives['fourier'](fourier, f_tar) * self.order_weights).mean(),
                                 self.weights['fourier'])
            if location_targets is not None:
                l_tar = location_targets[b, fg_indices]  # Tensor[num_pixels, 2]
                assert len(locations) == len(l_tar)
                add_to_loss_dict(losses, 'location', objectives['location'](locations, l_tar), self.weights['location'])
            if contour_targets is not None:
                add_to_loss_dict(losses, 'contour', objectives['contour'](contours, c_tar), self.weights['contour'])

                if self.refinement and self.refinement_iterations > 0:
                    if hires_contour_targets is None:
                        cc_tar = c_tar
                    else:
                        cc_tar = hires_contour_targets[b, fg_indices]  # Tensor[num_pixels, samples', 2]
                    for ref_con in all_refined_contours:
                        add_to_loss_dict(losses, 'refinement', objectives['refinement'](ref_con, cc_tar),
                                         self.weights['refinement'])

                if (uncertainty is not None and boxes.nelement() > 0 and box_targets is not None and
                        box_targets.nelement() > 0):
                    add_to_loss_dict(losses, 'uncertainty',
                                     objectives['uncertainty'](uncertainty, boxes.detach(), box_targets),
                                     self.weights['uncertainty'])

            if box_targets is not None:
                if objectives.get('iou') is not None:
                    add_to_loss_dict(losses, 'iou', objectives['iou'](boxes, box_targets), self.weights['iou'])
                if objectives.get('boxes') is not None:
                    add_to_loss_dict(losses, 'boxes', objectives['boxes'](boxes, box_targets), self.weights['boxes'])
        loss = reduce_loss_dict(losses, 1)
        closses = {}
        for k in losses.keys():
            if losses[k] is None:
                continue

            closses[k] = losses[k]
        return loss, closses

    # def generate_distri(self, data): # [B, N, 3]
    #     print("data shape: ", data.shape)
    #     mean = data.mean(-1, keepdim=True)
    #     std = data.std(-1, keepdim=True)
    #     # print(mean.shape, std.shape)
    #     f = 1. / (std * math.sqrt(2 * torch.pi))
    #     f = f * torch.exp(-((data - mean) ** 2 / (2 * std ** 2)))
    #     # add random dist
    #     f = f + torch.rand(f.shape, device=f.device)*0.001
    #     return f

    def generate_topo_label(self, n_ins):
        # r1: r2 = 1 : 2
        r1_values, r2_values = torch.tensor([1.0]), torch.tensor([2.0]) # [1]

        alph = (r1_values - r2_values) / (r1_values + r2_values)
        circumference = torch.pi * (r1_values + r2_values) * (
                1 + (3 * alph ** 2) / (10 + torch.sqrt(4 - 3 * alph ** 2)))

        # compute area
        area = torch.pi * r1_values * r2_values

        topo_rep = torch.cat([
            (r1_values / (r1_values + r2_values)).unsqueeze(-1),
            (r2_values / (r1_values + r2_values)).unsqueeze(-1),
            (4 * torch.pi * area / (circumference ** 2)).unsqueeze(-1)
        ], dim=-1)  # [1, 3]
        # topo_rep = topo_rep.unsqueeze(0) # [1, 1, 3]
        topo_rep = topo_rep.repeat(n_ins, 1) # [B, N, 3]
        # topo_rep = topo_rep + torch.rand(topo_rep.shape) * 0.001
        # print(topo_rep.shape)
        # topo_rep = self.generate_distri(topo_rep)
        return topo_rep

    def computer_topo2(self, cselected_contours):

        def polygon_area(vertices):
            vertices = torch.cat((vertices, vertices[:, 0:1]), dim=1)
            tri_area = 0.5 * torch.abs(
                torch.sum(vertices[:, :-1, 0] * vertices[:, 1:, 1] - vertices[:, :-1, 1] * vertices[:, 1:, 0], dim=1))
            area = torch.sum(tri_area)
            return area

        def polygon_perimeters(vertices):
            vertices = torch.cat((vertices, vertices[:, 0:1]), dim=1)
            edge_lengths = torch.norm(vertices[:, 1:] - vertices[:, :-1], dim=2)
            perimeters = torch.sum(edge_lengths, dim=1)
            return perimeters

        def reorder_vertices(vertices):
            centroid = torch.mean(vertices, dim=1)
            angles = torch.atan2(vertices[:, :, 1] - centroid[:, 1].unsqueeze(1),
                                 vertices[:, :, 0] - centroid[:, 0].unsqueeze(1))

            sorted_indices = torch.argsort(angles, dim=1)
            sorted_vertices = torch.gather(vertices, 1, sorted_indices.unsqueeze(2).expand(-1, -1, 2))

            return sorted_vertices

        topo_reps = []
        odistances = []

        for idx in range(len(cselected_contours)):
            icontours = cselected_contours[idx]  # [N_ins, 224, 2]
            if icontours.shape[0] == 0: # avoid none error
                continue

            # compute the center point of each instance
            centers = torch.cat([torch.mean(icontours[:, :, 0], dim=-1).unsqueeze(-1),
                                 torch.mean(icontours[:, :, 1], dim=-1).unsqueeze(-1)], dim=-1).unsqueeze(
                1)  # [N, 1, 2]
            centers = centers.repeat(1, icontours.shape[1], 1)  # [N, 224, 2]
            # computer the distance between center point and edge point
            odistance = torch.sqrt(
                (centers[:, :, 0] - icontours[:, :, 0]) ** 2 + (centers[:, :, 1] - icontours[:, :, 1]) ** 2
            )  # [N, 224]
            odistances.append(odistance)

            # compute area and perimeter
            vertices = reorder_vertices(icontours)
            area = polygon_area(vertices) # [N,]
            perimeter = polygon_perimeters(vertices)  # [N,]

            ratio = perimeter**2 / (4 * torch.pi * area + 1e-6) # [N,]
            topo_reps.append(ratio)

        odistances = torch.cat(odistances, dim=0)  # [N, 224]
        topo_reps = torch.cat(topo_reps, dim=0) # [N, ]

        return topo_reps, odistances # [N, 224], [N]

    def computer_topo(self, cselected_contours):

        topo_reps = []
        odistances = []
        for idx in range(len(cselected_contours)):
            icontours = cselected_contours[idx]  # [N_ins, 224, 2]
            if icontours.shape[0] == 0: # avoid none error
                continue

            # icontours = icontours * 100
            print(icontours.shape)
            # print(icontours)
            # print("#########################")

            all_icontours1 = icontours.repeat_interleave(repeats=icontours.shape[1], dim=1)  # [N_ins, 224*224, 2]
            all_icontours1 = all_icontours1.reshape(icontours.shape[0], icontours.shape[1], icontours.shape[1],
                                                    2)  # [N_ins, 224, 224, 2]

            all_icontours2 = icontours.repeat(1, icontours.shape[1], 1)  # [N_ins, 224*224, 2]
            all_icontours2 = all_icontours2.reshape(icontours.shape[0], icontours.shape[1], icontours.shape[1],
                                                    2)  # [N_ins, 224, 224, 2]

            # print(all_icontours1[:, :, :, 0] - all_icontours2[:, :, :, 0])
            # print(print(all_icontours1[:, :, :, 1] - all_icontours2[:, :, :, 1]))
            # print(all_icontours1[0,0,0,0], all_icontours2[0,0,0,0])
            # print(all_icontours1[0,0,0,0], all_icontours2[0,0,1,0])
            # print((all_icontours1[0,0,0, 0] - all_icontours2[0,0,1, 0]) ** 2 + (
            #         all_icontours1[0,0,0, 1] - all_icontours2[0,0,1, 1]) ** 2)
            # print(torch.sqrt((all_icontours1[0,0,1, 0] - all_icontours2[0,0,1, 0]) ** 2 + (
            #         all_icontours1[0,0,1, 1] - all_icontours2[0,0,1, 1]) ** 2))
            # assert 1 == 0
            distance_contours = torch.sqrt((all_icontours1[:, :, :, 0] - all_icontours2[:, :, :, 0]) ** 2 + (
                    all_icontours1[:, :, :, 1] - all_icontours2[:, :, :, 1]) ** 2) / 2.0
            # print(torch.sum(distance_contours))
            # assert  1 == 0
            # print("#####################################")
            # print(distance_contours.shape) # [N, 224, 224]
            # print(torch.max(distance_contours), torch.min(distance_contours))
            # print(distance_contours)
            # print(distance_contours.sort(dim=-1, descending=True)[1])
            distance_contourss = distance_contours.reshape(icontours.shape[0], -1)  # [N, -1]
            values_dis, indices = distance_contourss.sort(dim=-1, descending=True)
            # print("#####################################")
            # print(indices.shape) # [N, 224*224]
            # print(indices)
            top_idxs = indices[:, 0]  # [N, ]
            top_v = values_dis[:, 0]
            # print("#####################################")
            # print(top_idxs.shape)
            # print(top_idxs//icontours.shape[1], top_idxs%icontours.shape[1])
            r2_indexs = torch.tensor(
                [[idx, idy] for (idx, idy) in zip(top_idxs // icontours.shape[1], top_idxs % icontours.shape[1])])
            # print(r2_indexs.shape)  # [N_ins, 2] # [i, 0] and [i, 1] construct r2
            # print(r2_indexs)

            # get the points of r2
            r2_p1 = icontours[torch.arange(icontours.shape[0]), r2_indexs[:, 0], :]  # [N, 2]
            # print(r2_p1.shape) # [N, 2]
            # print(r2_p1)

            r2_p2 = icontours[torch.arange(icontours.shape[0]), r2_indexs[:, 1], :]  # [N, 2]
            # print(r2_p2.shape) # [N, 2]
            # print(r2_p2)

            # testdis = torch.sqrt((r2_p1[:, 0] - r2_p2[:, 0]) ** 2 + (
            #         r2_p1[:, 1] - r2_p2[:, 1]) ** 2)
            # print(testdis.shape) # [N]
            # print(testdis)
            # assert 1 == 0

            # get r2
            r2_values = top_v
            # print(r2_values.shape)
            # print(r2_values)  # [N, ]
            # print("#####################################")

            # other method to get r2
            # sdistance_contours = distance_contourss[torch.arange(icontours.shape[0]), top_idxs]
            # print(sdistance_contours.shape) #
            # print(sdistance_contours)

            # get r1
            vec_r2 = torch.cat([(r2_p2[:, 0] - r2_p1[:, 0]).unsqueeze(-1), (r2_p2[:, 1] - r2_p1[:, 1]).unsqueeze(-1)],
                               dim=-1)  # [N, 2]

            allp1 = all_icontours1  # [N, 224, 224, 2]
            allp2 = all_icontours2  # [N, 224, 224, 2]
            vec_all = torch.cat([(allp1[:, :, :, 0] - allp2[:, :, :, 0]).unsqueeze(-1),
                                 (allp1[:, :, :, 1] - allp2[:, :, :, 1]).unsqueeze(-1)], dim=-1)  # [N, 224, 224, 2]
            cvec_all = vec_all.reshape(icontours.shape[0], -1, 2)  # [N, 224*224, 2]

            # print("###########################")
            cosine_sim = torch.cosine_similarity(vec_r2.unsqueeze(1), cvec_all.unsqueeze(0), dim=-1).squeeze(0)
            angles = torch.acos(cosine_sim) * 180 / torch.pi
            tmask = torch.cat([torch.eye(icontours.shape[1]).unsqueeze(0) for _ in range(angles.shape[0])], dim=0).to(
                torch.bool).reshape(icontours.shape[0], -1)
            # print(angles.shape) # [N, 224*224]
            angles[tmask] = 0
            # print(angles)
            indices = torch.argmin(torch.abs(90 - angles.reshape(icontours.shape[0], -1)), dim=-1)  # [N]
            # print(indices//icontours.shape[1], indices%icontours.shape[1])
            r1_indexs = torch.tensor(
                [[idx, idy] for (idx, idy) in zip(indices // icontours.shape[1], indices % icontours.shape[1])])
            # print(r1_indexs.shape)  # [N_ins, 2] # [i, 0] and [i, 1] construct r2
            # print(r1_indexs)

            # get the points of r1
            r1_p1 = icontours[torch.arange(icontours.shape[0]), r1_indexs[:, 0], :]  # [N, 2]
            # print(r2_p1.shape) # [N, 2]
            # print(r2_p1)

            r1_p2 = icontours[torch.arange(icontours.shape[0]), r1_indexs[:, 1], :]  # [N, 2]
            # print(r1_p1, r1_p2)

            r1_values = torch.sqrt((r1_p1[:, 0] - r1_p2[:, 0]) ** 2 + (
                    r1_p1[:, 1] - r1_p2[:, 1]) ** 2) / 2.0
            # print(r1_values)

            # print(r2_values-r1_values)

            # compute Circumference
            alph = (r1_values - r2_values) / (r1_values + r2_values)
            circumference = torch.pi * (r1_values + r2_values) * (
                        1 + (3 * alph ** 2) / (10 + torch.sqrt(4 - 3 * alph ** 2)))

            # compute area
            area = torch.pi * r1_values * r2_values

            # print(circumference)
            #
            # print(area)
            # print(r1_values/(r1_values+r2_values)) # [N]
            # print(r2_values/(r1_values+r2_values))
            # print(4*torch.pi*area/(circumference**2))
            # assert 1 == 0
            topo_rep = torch.cat([
                (r1_values / (r1_values + r2_values)).unsqueeze(-1),
                (r2_values / (r1_values + r2_values)).unsqueeze(-1),
                (4 * torch.pi * area / (circumference ** 2)).unsqueeze(-1)
            ], dim=-1)  # [N, 3]
            topo_reps.append(topo_rep)

            # compute the center point of each instance
            centers = torch.cat([torch.mean(icontours[:, :, 0], dim=-1).unsqueeze(-1), torch.mean(icontours[:, :, 1], dim=-1).unsqueeze(-1)], dim=-1).unsqueeze(1) # [N, 1, 2]
            centers = centers.repeat(1, icontours.shape[1], 1) # [N, 224, 2]
            # computer the distance between center point and edge point
            odistance = torch.sqrt(
                (centers[:, :, 0]-icontours[:, :, 0])**2 + (centers[:, :, 1]-icontours[:, :, 1])**2
            ) # [N, 224]
            odistances.append(odistance)

        topo_reps = torch.cat(topo_reps, dim=0)  # [N, 3]
        # print("topo_reps shape: ", topo_reps.shape)
        # print(topo_reps)
        # topo_distri = self.generate_distri(topo_reps)
        # topo_distri = self.generate_distri(topo_rep.reshape(topo_reps.shape[0], -1))
        # print(topo_distri.shape)
        # print(topo_distri)
        # assert 1 == 0
        odistances = torch.cat(odistances, dim=0) # [N, 224]

        return topo_reps, odistances

    def forward(
            self,
            inputs,
            targets: Dict[str, Tensor] = None,
            nms=True,
            # freq_guild=None,
            **kwargs
    ):
        # Presets
        original_size = inputs.shape[-2:] # [H, W]

        # Core
        # print(inputs.shape)
        scores, locations, refinement, fourier, uncertainty, guild_seg, features = self.core(inputs)
        # print("scores, locations, refinement, fourier, uncertainty")
        # torch.Size([2, 1, 128, 128])
        # torch.Size([2, 2, 128, 128])
        # torch.Size([2, 12, 512, 512])
        # torch.Size([2, 100, 128, 128])
        # torch.Size([2, 4, 128, 128])
        # print(scores.shape, locations.shape, refinement.shape, fourier.shape, uncertainty.shape)
        # assert 1 == 0
        # if scores.shape[-2:] != original_size:
        #     # print("resize score")
        #     scores = _equal_size(scores, inputs)
        # if locations.shape[-2:] != original_size:
        #     # print("resize locations")
        #     locations = _equal_size(locations, inputs)
        # if refinement.shape[-2:] != original_size:
        #     # print("resize refinement")
        #     refinement = _equal_size(refinement, inputs)
        # if fourier.shape[-2:] != original_size:
        #     # print("resize fourier")
        #     fourier = _equal_size(fourier, inputs)
        # if uncertainty.shape[-2:] != original_size:
        #     # print("resize uncertainty")
        #     uncertainty = _equal_size(uncertainty, inputs)

        # Scores
        raw_scores = scores
        score_bounds = kwargs.get('scores_lower_bound'), kwargs.get('scores_upper_bound')
        if self.score_channels == 1:
            scores = _apply_score_bounds(torch.sigmoid(scores), *score_bounds)
            classes = torch.squeeze((scores > self.score_thresh).long(), 1)
        elif self.score_channels == 2:
            scores = _apply_score_bounds(F.softmax(scores, dim=1)[:, 1:2], *score_bounds)
            classes = torch.squeeze((scores > self.score_thresh).long(), 1)
        elif self.score_channels > 2:
            scores = _apply_score_bounds(F.softmax(scores, dim=1), *score_bounds)
            classes = torch.argmax(scores, dim=1).long()
        else:
            raise ValueError

        actual_size = fourier.shape[-2:]
        n, c, h, w = fourier.shape
        if self.functional:
            fourier = fourier.view((n, c // 2, 2, h, w))
        else: # use this
            fourier = fourier.view((n, c // 4, 4, h, w)) # [n, order, 4, h, w]

        # Maybe apply changed order
        if self.order < self.core.order:
            fourier = fourier[:, :self.order]

        # Fetch sampling and labels
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            sampling = targets.get('sampling') # [n, sampling_num(224)]
            labels = targets['labels'] # [n, H, W]
            fgimg_mask = targets.get('fg_mask', None)
            bmask = targets.get("bmask", None)

        else:
            sampling = None
            labels = classes.detach()
            fgimg_mask = None
            bmask = None


        # through F.max_pool2d
        labels = downsample_labels(labels[:, None], actual_size)[:, 0]  # [n, h, w]
        # labels = downsample_labels(labels[:, None], actual_size)  # [n, 1, h, w]

        if fgimg_mask is not None:
            fgimg_mask = downsample_labels(fgimg_mask[:, None], actual_size)[:, 0] # [n, h, w]

        # Locations
        # raw_locations = locations.detach()
        locations = rel_location2abs_location(locations, cache=self._rel_location2abs_location_cache)

        # Extract proposals
        fg_mask = labels > 0 # remove bg pixel

        # ban white-trending area of img
        if fgimg_mask is not None and self.ignore_bg:
            assert len(fgimg_mask.shape()) == 2
            fg_mask &= (fgimg_mask==1)
            # assert 1 == 0

        if self.certainty_thresh is not None and uncertainty is not None:
            fg_mask &= uncertainty.mean(1) < (1 - self.certainty_thresh)
        # we should concentrate on how to improve the item: uncertainty

        b, y, x = torch.where(fg_mask)
        selected_fourier = fourier[b, :, :, y, x]  # Tensor[-1, order, 4]
        selected_locations = locations[b, :, y, x]  # Tensor[-1, 2]
        selected_classes = classes[b, y, x]

        if self.score_channels in (1, 2):
            selected_scores = scores[b, 0, y, x]  # Tensor[-1]
        elif self.score_channels > 2:
            selected_scores = scores[b, selected_classes, y, x]  # Tensor[-1]
        else:
            raise ValueError

        selected_uncertainties = None
        if uncertainty is not None:
            selected_uncertainties = uncertainty[b, :, y, x] # [-1, 4]

        if sampling is not None:
            sampling = sampling[b]

        # Convert to pixel space
        selected_contour_proposals, sampling = fouriers2contours(selected_fourier, selected_locations,
                                                                 samples=self.samples, sampling=sampling,
                                                                 cache=self._fourier2contour_cache)

        # Rescale in case of multi-scale
        selected_contour_proposals = scale_contours(actual_size=actual_size, original_size=original_size,
                                                    contours=selected_contour_proposals)
        selected_fourier, selected_locations = scale_fourier(actual_size=actual_size, original_size=original_size,
                                                             fourier=selected_fourier, location=selected_locations)

        if self.refinement and self.refinement_iterations > 0:
            num_loops = self.refinement_iterations
            # if self.training and num_loops > 1:  # Note: Changed to fixed num
            #     num_loops = torch.randint(low=1, high=num_loops + 1, size=())
            selected_contours, all_ref_selected_contours = local_refinement(
                selected_contour_proposals, refinement, num_loops=num_loops, num_buckets=self.core.refinement_buckets,
                original_size=original_size, sampling=sampling, b=b
            )
        else:
            selected_contours = selected_contour_proposals
            all_ref_selected_contours = [selected_contours]

        for sel_con in all_ref_selected_contours:
            sel_con[..., 0].clamp_(0, original_size[1] - 1)
            sel_con[..., 1].clamp_(0, original_size[0] - 1)

        # Bounding boxes
        if selected_contours.numel() > 0:
            selected_boxes = torch.cat((selected_contours.min(1).values,
                                        selected_contours.max(1).values), 1)  # 43.3 µs ± 290 ns for Tensor[2203, 32, 2]
        else:
            selected_boxes = torch.empty((0, 4), device=selected_contours.device)

        # Loss
        loss, losses = None, None
        if self.training or targets is not None:
            loss, losses = self.compute_loss(
                uncertainty=selected_uncertainties,
                fourier=selected_fourier,
                locations=selected_locations,
                contours=selected_contour_proposals,
                all_refined_contours=all_ref_selected_contours,
                refined_contours=selected_contours,
                boxes=selected_boxes,
                raw_scores=raw_scores,
                targets=targets,
                labels=labels,
                fg_masks=fg_mask,
                sampling=sampling,
                b=b
            )

            # Dummy loss
            # print("Dummy loss")
            # print(dummy_loss(raw_scores, locations, refinement, uncertainty, fourier))
            # loss = loss + dummy_loss(raw_scores, locations, refinement, uncertainty, fourier)

            if self.use_guild == 'seg':
                assert guild_seg is not None
                seg_gt = targets.get('low_freq_guild', None)
                assert seg_gt is not None
                # print(seg_gt.shape, bmask.shape)
                assert bmask is not None
                seg_gt += bmask
                seg_gt = torch.clip(seg_gt, max=1.0, min=0.0)

                # print(seg_gt.shape, guild_seg.shape)
                # loss_seg = F.l1_loss(guild_seg, seg_gt)
                if isinstance(guild_seg, dict):
                    loss_seg = 0.0
                    for k in guild_seg.keys():
                        guild_segi = _equal_size(guild_seg[k], inputs) # [B, 1, h, w] -> [B, 1, H, W]
                        loss_segi = F.binary_cross_entropy(guild_segi, seg_gt)
                        loss_seg = loss_seg + loss_segi
                        losses['loss_seg_'+str(k)] = loss_segi
                else:

                    loss_seg = F.binary_cross_entropy(guild_seg, seg_gt)
                    losses['loss_seg'] = loss_seg
                # assert 1 == 0
                loss = loss + loss_seg

                if self.use_contr:
                    loss_contr = 0.0
                    for k in ['0']:
                        assert k in features
                        q1 = features[k] # [B, c, h, w]
                        q2 = _equal_size(seg_gt, q1) # [B, 1, H, W]

                        id1 = q2 > 0.7 # positive, [B, 1, h, w]
                        id2 = q2 < 0.3 # negative  [B, 1, h, w]
                        for i in range(q1.shape[0]):
                            id1i = id1[i][0] # [h, w]
                            id2i = id2[i][0] # [h, w]
                            q1i = q1[i] # [c, h, w]

                            positive_features = q1i[:, id1i].permute(1, 0) # [c, N1] -> [N1, c]
                            negative_features = q1i[:, id2i].permute(1, 0) # [c, N2] -> [N2, c]

                            anchor_indices = torch.randint(0, positive_features.shape[0], (positive_features.shape[0], ))
                            anchors = positive_features[anchor_indices]

                            positive_indices = torch.randint(0, positive_features.shape[0], (positive_features.shape[0], ))
                            positives = positive_features[positive_indices]

                            negative_indices = torch.randint(0, negative_features.shape[0], (positive_features.shape[0], ))
                            negatives = negative_features[negative_indices]

                            loss_contr += F.triplet_margin_loss(anchors, positives, negatives, margin=2.0)

                    losses['loss_contrastive'] = loss_contr

                    loss = loss + loss_contr



        # assert 1 == 0

        # losses['overall_loss'] = loss
        # print(losses.keys()) # dict_keys(['location', 'contour', 'score', 'refinement', 'iou', 'uncertainty', 'overall_loss'])

        # Optional offsets (loss calc etc. not affected)
        offsets: Union[Tensor, None] = kwargs.pop('offsets', None)
        if offsets is not None:
            offsets = offsets[b]
            offsets_ = offsets[:, None]
            selected_contours += offsets_  # assuming xy format
            selected_contour_proposals += offsets_  # assuming xy format
            selected_boxes += offsets.repeat((1,) * (offsets.ndim - 1) + (2,))  # assuming xy format
            selected_locations += offsets  # assuming xy format
        # print(selected_contours.shape) # [N_ins, 224, 2]
        # print(selected_contours[0])


        outputs = OrderedDict(
            contours=selected_contours,
            boxes=selected_boxes,
            scores=selected_scores,
            classes=selected_classes,
            locations=selected_locations,
            fourier=selected_fourier,
            contour_proposals=selected_contour_proposals,
            box_uncertainties=selected_uncertainties,
        )
        # print(selected_scores.shape)


        outputs = resolve_batch_index(outputs, inputs.shape[0], b=b)
        # print(type(outputs))
        # cselected_contours = outputs['contours']
        # print(type(cselected_contours)) # list
        # print(len(cselected_contours)) # B
        # print(cselected_contours[0].shape) # [B, N_ins, 224, 2]
        # assert len(cselected_contours) == b
        # print(outputs['contours'][0][0].shape)
        # print(outputs['contours'][0][0])
        # assert 1 == 0
        if self.training and self.use_topo_loss:
            topo_distri, odistances = self.computer_topo2(outputs['contours'])
            # assert 1 == 0
            pred_ = topo_distri.to(selected_contours.device) # [N, 3]
            # print(pred_)
            label_ = self.generate_topo_label(pred_.shape[0]).to(selected_contours.device)
            # print(label_)

            loss_topo = F.kl_div(F.log_softmax(pred_, dim=-1), F.softmax(label_, dim=-1), reduction='batchmean')
            losses['loss_topo'] = loss_topo
            loss = loss + loss_topo
            # print(loss_topo)
            # assert 1 == 0

        if self.training and not self.full_detail:
            return OrderedDict({
                'loss': loss,
                'losses': losses,
            })

        if not self.training and nms:
            if self.uncertainty_nms and outputs['box_uncertainties'] is not None:
                nms_weights = [s * (1. - u.mean(1)) for s, u in zip(outputs['scores'], outputs['box_uncertainties'])]
            else:
                nms_weights = outputs['scores']
            keep_indices: list = batched_box_nmsi(outputs['boxes'], nms_weights, self.nms_thresh)
            outputs = resolve_keep_indices(outputs, keep_indices)

        if loss is not None:
            outputs['loss'] = loss
            outputs['losses'] = losses

        if guild_seg is not None:
            if isinstance(guild_seg, dict):
                assert '0' in guild_seg.keys()
                # print(guild_seg.keys())
                guild_seg = _equal_size(guild_seg['0'], inputs)

            print(guild_seg.shape)
            outputs['guild_seg'] = guild_seg.detach().cpu().squeeze(1).numpy()  # [B, H, W]

        return outputs

class CpnResNeXt101UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNeXt101UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()


class CpnU22(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=U22(in_channels, 0, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()


class CpnResUNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=ResUNet(in_channels, 0, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnSlimU22(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=SlimU22(in_channels, 0, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnWideU22(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=WideU22(in_channels, 0, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnResNet101UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNet101UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnResNeXt50UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNeXt50UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnResNet50UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNet50UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnResNet34UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNet34UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnResNet18UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNet18UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnResNet18FPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=ResNet18FPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnResNet34FPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=ResNet34FPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnResNet50FPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=ResNet50FPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnResNet101FPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=ResNet101FPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnResNeXt50FPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=ResNeXt50FPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnWideResNet50FPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=WideResNet50FPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnWideResNet101FPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=WideResNet101FPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnMobileNetV3SmallFPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=MobileNetV3SmallFPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnMobileNetV3LargeFPN(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs: dict = None,
            **kwargs
    ):
        super().__init__(
            backbone=MobileNetV3LargeFPN(in_channels, **(backbone_kwargs or {})),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnConvNeXtSmallUNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ConvNeXtSmallUNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()

class CpnResNet101UNet(CPN):
    def __init__(
            self,
            in_channels: int,
            order: int = 5,
            nms_thresh: float = .2,
            score_thresh: float = .9,
            samples: int = 32,
            classes: int = 2,
            refinement: bool = True,
            refinement_iterations: int = 4,
            refinement_margin: float = 3.,
            refinement_buckets: int = 1,
            backbone_kwargs=None,
            **kwargs
    ):
        backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
        super().__init__(
            backbone=ResNet101UNet(in_channels, 0, **backbone_kwargs),
            order=order,
            nms_thresh=nms_thresh,
            score_thresh=score_thresh,
            samples=samples,
            classes=classes,
            refinement=refinement,
            refinement_iterations=refinement_iterations,
            refinement_margin=refinement_margin,
            refinement_buckets=refinement_buckets,
            **kwargs
        )
        self.save_hyperparameters()






