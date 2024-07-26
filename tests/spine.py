import warnings
import numpy as np
import inspect
import copy
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Union, List, Tuple, Any, Dict as TDict, Iterator, Type, Callable, Iterable, Sequence
from torch import Tensor
from torch.hub import load_state_dict_from_url
import hashlib
import json
from tqdm import tqdm
from os.path import join, isfile, splitext
from os import makedirs
import pynvml as nv
from cv2 import getGaussianKernel
import h5py
from collections import OrderedDict
import re
import sys
from itertools import product
from inspect import currentframe, signature
from shutil import copy2
from PIL import Image
from io import BytesIO
from base64 import b64encode, b64decode
from glob import glob
import math


def resolve_pretrained(pretrained, state_dict_mapper=None, **kwargs):
    if isinstance(pretrained, str):
        if isfile(pretrained):
            state_dict = torch.load(pretrained)
        else:
            state_dict = load_state_dict_from_url(pretrained)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if '.pytorch.org' in pretrained:
            if state_dict_mapper is not None:
                state_dict = state_dict_mapper(state_dict=state_dict, **kwargs)
    else:
        raise ValueError('There is no default set of weights for this model. '
                         'Please specify a URL or filename using the `pretrained` argument.')
    return state_dict

def update_dict_(dst, src, override=False, keys: Union[List[str], Tuple[str]] = None):
    for k, v in src.items():
        if keys is not None and k not in keys:
            continue
        if override or k not in dst:
            dst[k] = v

def replace_ndim(s: Union[str, type, Callable], dim: int, allowed_dims=(1, 2, 3)):
    """Replace ndim.

    Replaces dimension statement of ``string``or ``type``.

    Notes:
        - Dimensions are expected to be at the end of the type name.
        - If there is no dimension statement, nothing is changed.

    Examples:
        >>> replace_ndim('BatchNorm2d', 3)
        'BatchNorm3d'
        >>> replace_ndim(nn.BatchNorm2d, 3)
        torch.nn.modules.batchnorm.BatchNorm3d
        >>> replace_ndim(nn.GroupNorm, 3)
        torch.nn.modules.normalization.GroupNorm
        >>> replace_ndim(F.conv2d, 3)
        <function torch._VariableFunctionsClass.conv3d>

    Args:
        s: String or type.
        dim: Desired dimension.
        allowed_dims: Allowed dimensions to look for.

    Returns:
        Input with replaced dimension.
    """
    if isinstance(s, str) and dim in allowed_dims:
        return re.sub(f"[1-3]d$", f'{int(dim)}d', s)
    elif isinstance(s, type) or callable(s):
        return getattr(sys.modules[s.__module__], replace_ndim(s.__name__, dim))
    return s

def lookup_nn(item: str, *a, src=None, call=True, inplace=True, nd=None, **kw):
    """

    Examples:
        >>> lookup_nn('batchnorm2d', 32)
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> lookup_nn(torch.nn.BatchNorm2d, 32)
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> lookup_nn('batchnorm2d', num_features=32)
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> lookup_nn('tanh')
            Tanh()
        >>> lookup_nn('tanh', call=False)
            torch.nn.modules.activation.Tanh
        >>> lookup_nn('relu')
            ReLU(inplace=True)
        >>> lookup_nn('relu', inplace=False)
            ReLU()
        >>> # Dict notation to contain all keyword arguments for calling in `item`. Always called once.
        ... lookup_nn(dict(relu=dict(inplace=True)), call=False)
            ReLU(inplace=True)
        >>> lookup_nn({'NormProxy': {'norm': 'GroupNorm', 'num_groups': 32}}, call=False)
            NormProxy(GroupNorm, kwargs={'num_groups': 32})
        >>> lookup_nn({'NormProxy': {'norm': 'GroupNorm', 'num_groups': 32}}, 32, call=True)
            GroupNorm(32, 32, eps=1e-05, affine=True)

    Args:
        item: Lookup item. None is equivalent to `identity`.
        *a: Arguments passed to item if called.
        src: Lookup source.
        call: Whether to call item.
        inplace: Default setting for items that take an `inplace` argument when called.
            As default is True, `lookup_nn('relu')` returns a ReLu instance with `inplace=True`.
        nd: If set, replace dimension statement (e.g. '2d' in nn.Conv2d) with ``nd``.
        **kw: Keyword arguments passed to item when it is called.

    Returns:
        Looked up item.
    """
    src = (nn, )
    if isinstance(item, tuple):
        if len(item) == 1:
            item, = item
        elif len(item) == 2:
            item, _kw = item
            kw.update(_kw)
        else:
            raise ValueError('Allowed formats for item: (item,) or (item, kwargs).')
    if item is None:
        v = nn.Identity
    elif isinstance(item, str):
        l_item = item.lower()
        if nd is not None:
            l_item = replace_ndim(l_item, nd)
        if not isinstance(src, (list, tuple)):
            src = src,
        v = None
        for src_ in src:
            print(src_)
            try:
                v = next((getattr(src_, i) for i in dir(src_) if i.lower() == l_item))
            except StopIteration:
                continue
            break
        if v is None:
            raise ValueError(f'Could not find `{item}` in {src}.')
    elif isinstance(item, nn.Module):
        return item
    elif isinstance(item, dict):
        assert len(item) == 1
        key, = item
        val = item[key]
        assert isinstance(val, dict)
        # print(item)
        # assert 1 == 0
        cls = lookup_nn(key, src=src, call=False, inplace=inplace, nd=nd)
        if issubclass(cls, nn.modules.loss._WeightedLoss):  # allows weight to be passed as lists (common use case)
            if 'weight' in val and not isinstance(val['weight'], Tensor):
                val['weight'] = torch.as_tensor(val['weight'])
        v = cls(**val)
    elif isinstance(item, type) and nd is not None:
        v = replace_ndim(item, nd)
    else:
        v = item
    if call:
        kwargs = {'inplace': inplace} if 'inplace' in inspect.getfullargspec(v).args else {}
        kwargs.update(kw)
        v = v(*a, **kwargs)
    return v

class NormProxy:
    def __init__(self, norm, **kwargs):
        """Norm Proxy.

        Examples:
            >>> GroupNorm = NormProxy('groupnorm', num_groups=32)
            ... GroupNorm(3)
            GroupNorm(32, 3, eps=1e-05, affine=True)
            >>> GroupNorm = NormProxy(nn.GroupNorm, num_groups=32)
            ... GroupNorm(3)
            GroupNorm(32, 3, eps=1e-05, affine=True)
            >>> BatchNorm2d = NormProxy('batchnorm2d', momentum=.2)
            ... BatchNorm2d(3)
            BatchNorm2d(3, eps=1e-05, momentum=0.2, affine=True, track_running_stats=True)
            >>> BatchNorm2d = NormProxy(nn.BatchNorm2d, momentum=.2)
            ... BatchNorm2d(3)
            BatchNorm2d(3, eps=1e-05, momentum=0.2, affine=True, track_running_stats=True)

        Args:
            norm: Norm class or name.
            **kwargs: Keyword arguments.
        """
        self.norm = norm
        self.kwargs = kwargs

    def __call__(self, num_channels):
        Norm = lookup_nn(self.norm, call=False)
        kwargs = dict(self.kwargs)
        args = inspect.getfullargspec(Norm).args
        if 'num_features' in args:
            kwargs['num_features'] = num_channels
        elif 'num_channels' in args:
            kwargs['num_channels'] = num_channels
        return Norm(**kwargs)

    def __repr__(self):
        return f'NormProxy({self.norm}, kwargs={self.kwargs})'

    __str__ = __repr__

def model2dict(model: 'nn.Module'):
    return dict(
        model=model.__class__.__name__,
        kwargs=dict(model.hparams),
    )

def random_seed(seed, backends=False, deterministic_torch=True):
    """Set random seed.

    Set random seed to ``random``, ``np.random``, ``torch.backends.cudnn`` and ``torch.manual_seed``.
    Also advise torch to use deterministic algorithms.

    References:
        https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        seed: Random seed.
        backends: Whether to also adapt backends. If set True cuDNN's benchmark feature is disabled. This
            causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
            Also the selected algorithm is set to run deterministically.
        deterministic_torch: Whether to set PyTorch operations to behave deterministically.

    """
    from torch import manual_seed
    from torch.backends import cudnn
    import random
    random.seed(seed)
    manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
#     if backends:
#         cudnn.deterministic = True
#         cudnn.benchmark = False
#     if deterministic_torch and 'use_deterministic_algorithms' in dir(torch):
#         torch.use_deterministic_algorithms(True)

def get_nd_conv(dim: int):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return getattr(nn, 'Conv%dd' % dim)


def get_nd_max_pool(dim: int):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return getattr(nn, 'MaxPool%dd' % dim)


def get_nd_batchnorm(dim: int):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return getattr(nn, 'BatchNorm%dd' % dim)


def get_nd_dropout(dim: int):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return getattr(nn, 'Dropout%dd' % dim)


def get_nd_linear(dim: int):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return ['', 'bi', 'tri'][dim - 1] + 'linear'

# util.coordconv

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))

class CoordConv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        x = self.relu(self.bn(x))
        return x

# ops.commons
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import List

def equal_size(x, reference, mode='bilinear', align_corners=False):
    if reference.shape[2:] != x.shape[2:]:  # 337 ns
        # bilinear: 3.79 ms for (128, 128) to (512, 512)
        # bicubic: 11.5 ms for (128, 128) to (512, 512)
        x = F.interpolate(x, reference.shape[2:],
                          mode=mode, align_corners=align_corners)
    return x

def _apply_score_bounds(scores, scores_lower_bound, scores_upper_bound):
    if scores_upper_bound is not None:
        assert scores_upper_bound.ndim >= 4, (f'Please make sure scores_upper_bound comes in NCHW format: '
                                              f'{scores_upper_bound.shape}')
        assert scores_upper_bound.dtype.is_floating_point, (f'Please make sure to pass scores_upper_bound as float '
                                                            f'instead of {scores_upper_bound.dtype}')
        scores = torch.minimum(scores, equal_size(scores_upper_bound, scores))
    if scores_lower_bound is not None:
        assert scores_lower_bound.ndim >= 4, (f'Please make sure scores_upper_bound comes in NCHW format: '
                                              f'{scores_lower_bound.shape}')
        assert scores_lower_bound.dtype.is_floating_point, (f'Please make sure to pass scores_upper_bound as float '
                                                            f'instead of {scores_lower_bound.dtype}')
        scores = torch.maximum(scores, equal_size(scores_lower_bound, scores))
    return scores

def process_scores(scores, score_channels, score_thresh, scores_lower_bound, scores_upper_bound):
    score_bounds = scores_lower_bound, scores_upper_bound
    if score_channels == 1:
        scores = _apply_score_bounds(torch.sigmoid(scores), *score_bounds)
        classes = torch.squeeze((scores > score_thresh).long(), 1)
    elif score_channels == 2:
        scores = _apply_score_bounds(F.softmax(scores, dim=1)[:, 1:2], *score_bounds)
        classes = torch.squeeze((scores > score_thresh).long(), 1)
    elif score_channels > 2:
        scores = _apply_score_bounds(F.softmax(scores, dim=1), *score_bounds)
        classes = torch.argmax(scores, dim=1).long()
    else:
        raise ValueError
    return scores, classes

def downsample_labels(inputs, size: List[int]):
    """

    Down-sample via max-pooling and interpolation

    Notes:
        - Downsampling can lead to loss of labeled instances, both during max pooling and interpolation.
        - Typical timing: 0.08106 ms for 256x256

    Args:
        inputs: Label Tensor to resize. Shape (n, c, h, w)
        size: Tuple containing target height and width.

    Returns:

    """
    sizeh, sizew = size  # for torchscript
    if inputs.shape[-2:] == (sizeh, sizew):
        return inputs
    if inputs.dtype != torch.float32:
        inputs = inputs.float()
    h, w = inputs.shape[-2:]
    th, tw = size
    k = h // th, w // tw
    r = F.max_pool2d(inputs, k, k)
    if r.shape[-2:] != (sizeh, sizew):
        r = F.interpolate(r, size, mode='nearest')
    return r

def strided_upsampling2d(x, factor=2, const=0):
    """Strided upsampling.

    Upsample by inserting rows and columns filled with ``constant``.

    Args:
        x: Tensor[n, c, h, w].
        factor: Upsampling factor.
        const: Constant used to fill inserted rows and columns.

    Returns:
        Tensor[n, c, h*factor, w*factor].
    """
    n, c, h, w = x.shape
    x_ = torch.zeros((n, c, h * factor, w * factor), dtype=x.dtype, device=x.device)
    if const != 0:
        x_.fill_(const)
    x_[..., ::factor, ::factor] = x
    return x_

def interpolate_vector(v, size, **kwargs):
    """Interpolate vector.

    Args:
        v: Vector as ``Tensor[d]``.
        size: Target size.
        **kwargs: Keyword arguments for ``F.interpolate``

    Returns:

    """
    return torch.squeeze(torch.squeeze(
        F.interpolate(v[None, None], size, **kwargs), 0
    ), 0)

def pad_to_size(v, size, return_pad=False, **kwargs):
    """Pad tp size.

    Applies padding to end of each dimension.

    Args:
        v: Input Tensor.
        size: Size tuple. Last element corresponds to last dimension of input `v`.
        return_pad: Whether to return padding values.
        **kwargs: Additional keyword arguments for `F.pad`.

    Returns:
        Padded Tensor.
    """
    pad = []
    for a, b in zip(size, v.shape[-len(size):]):
        pad += [max(0, a - b), 0]
    if any(pad):
        v = F.pad(v, pad[::-1], **kwargs)
    if return_pad:
        return v, pad
    return v


def pad_to_div(v, div=32, nd=2, return_pad=False, **kwargs):
    """Pad to div.

    Applies padding to input Tensor to make it divisible by `div`.

    Args:
        v: Input Tensor.
        div: Div tuple. If single integer, `nd` is used to define number of dimensions to pad.
        nd: Number of dimensions to pad. Only used if `div` is not a tuple or list.
        return_pad: Whether to return padding values.
        **kwargs: Additional keyword arguments for `F.pad`.

    Returns:
        Padded Tensor.
    """
    if not isinstance(div, (tuple, list)):
        div = (div,) * nd
    size = [(i // d + bool(i % d)) * d for i, d in zip(v.shape[-len(div):], div)]
    return pad_to_size(v, size, return_pad=return_pad, **kwargs)


def spatial_mean(x, keepdim=False):
    spatial = tuple(range(2, x.ndim))
    return torch.mean(x, spatial, keepdim=keepdim)

# model.commons
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, tanh, sigmoid
from torchvision import transforms as trans
from torch.nn.common_types import _size_2_t
# from ..util.util import lookup_nn, tensor_to, ensure_num_tuple, get_nd_conv
# from ..ops.commons import split_spatially, minibatch_std_layer
from typing import Type, Union
from functools import partial


def _ni_3d(nd):
    if nd != 2:
        raise NotImplementedError('The `nd` option is not yet available for this model.')

class ConvNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, norm_layer=nn.BatchNorm2d,
                 nd=2, **kwargs):
        """ConvNorm.

        Just a convolution and a normalization layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size.
            padding: Padding.
            stride: Stride.
            norm_layer: Normalization layer (e.g. ``nn.BatchNorm2d``).
            **kwargs: Additional keyword arguments.
        """
        Conv = get_nd_conv(nd)
        #         Norm = lookup_nn(norm_layer, nd=nd, call=False)
        if 'batch' in str(norm_layer).lower():
            Norm = lookup_nn(nn.BatchNorm2d, nd=nd, call=False)
        elif 'group' in str(norm_layer).lower():
            Norm = NormProxy(nn.GroupNorm, num_groups=32)

        super().__init__(
            Conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, **kwargs),
            Norm(out_channels),
        )


class ConvNormRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, norm_layer=nn.BatchNorm2d,
                 activation='relu', nd=2, **kwargs):
        """ConvNormReLU.

        Just a convolution, normalization layer and an activation.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size.
            padding: Padding.
            stride: Stride.
            norm_layer: Normalization layer (e.g. ``nn.BatchNorm2d``).
            activation: Activation function. (e.g. ``nn.ReLU``, ``'relu'``)
            **kwargs: Additional keyword arguments.
        """
        Conv = get_nd_conv(nd)
        #         Norm = lookup_nn(norm_layer, nd=nd, call=False)
        if 'batch' in str(norm_layer).lower():
            Norm = lookup_nn(nn.BatchNorm2d, nd=nd, call=False)
        elif 'group' in str(norm_layer).lower():
            Norm = NormProxy(nn.GroupNorm, num_groups=32)

        super().__init__(
            Conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, **kwargs),
            Norm(out_channels),
            lookup_nn(activation)
        )


class TwoConvNormRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, mid_channels=None,
                 norm_layer=nn.BatchNorm2d, activation='relu', nd=2, **kwargs):
        """TwoConvNormReLU.

        A sequence of conv, norm, activation, conv, norm, activation.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size.
            padding: Padding.
            stride: Stride.
            mid_channels: Mid-channels. Default: Same as ``out_channels``.
            norm_layer: Normalization layer (e.g. ``nn.BatchNorm2d``).
            activation: Activation function. (e.g. ``nn.ReLU``, ``'relu'``)
            **kwargs: Additional keyword arguments.
        """
        Conv = get_nd_conv(nd)
        #         Norm = lookup_nn(norm_layer, nd=nd, call=False)
        if 'batch' in str(norm_layer).lower():
            Norm = lookup_nn(nn.BatchNorm2d, nd=nd, call=False)
        elif 'group' in str(norm_layer).lower():
            Norm = NormProxy(nn.GroupNorm, num_groups=32)

        if mid_channels is None:
            mid_channels = out_channels
        super().__init__(
            Conv(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, stride=stride, **kwargs),
            Norm(mid_channels),
            lookup_nn(activation),
            Conv(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, **kwargs),
            Norm(out_channels),
            lookup_nn(activation)
        )

class TwoConvNormLeaky(TwoConvNormRelu):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, mid_channels=None,
                 norm_layer=nn.BatchNorm2d, nd=2, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                         mid_channels=mid_channels, norm_layer=norm_layer, activation='leakyrelu', nd=nd, **kwargs)


class ScaledX(nn.Module):
    def __init__(self, fn, factor, shift=0.):
        super().__init__()
        self.factor = factor
        self.shift = shift
        self.fn = fn

    def forward(self, inputs: Tensor) -> Tensor:
        return self.fn(inputs) * self.factor + self.shift

    def extra_repr(self) -> str:
        return 'factor={}, shift={}'.format(self.factor, self.shift)

class _ResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            block,
            activation='ReLU',
            stride=1,
            downsample=None,
            norm_layer='BatchNorm2d',
            nd=2,
    ) -> None:
        """ResBlock.

        Typical ResBlock with variable kernel size and an included mapping of the identity to correct dimensions.

        References:
            https://arxiv.org/abs/1512.03385

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size.
            padding: Padding.
            norm_layer: Norm layer.
            activation: Activation.
            stride: Stride.
            downsample: Downsample module that maps identity to correct dimensions. Default is an optionally strided
                1x1 Conv2d with BatchNorm2d, as per He et al. (2015) (`3.3. Network Architectures`, `Residual Network`,
                "option (B)").
            nd: Number of spatial dimensions.
        """
        super().__init__()
        downsample = downsample or partial(ConvNorm, nd=nd, norm_layer=norm_layer)
        if in_channels != out_channels or stride != 1:
            self.downsample = downsample(in_channels, out_channels, 1, stride=stride, bias=False, padding=0)
        else:
            self.downsample = nn.Identity()
        self.block = block
        self.activation = lookup_nn(activation)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)
        out = self.block(x)
        out += identity
        return self.activation(out)


class ResBlock(_ResBlock):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer='BatchNorm2d',
            activation='ReLU',
            stride=1,
            downsample=None,
            nd=2,
            **kwargs
    ) -> None:
        """ResBlock.

        Typical ResBlock with variable kernel size and an included mapping of the identity to correct dimensions.

        References:
            - https://doi.org/10.1109/CVPR.2016.90

        Notes:
            - Similar to ``torchvision.models.resnet.BasicBlock``, with different interface and defaults.
            - Consistent with standard signature ``in_channels, out_channels, kernel_size, ...``.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size.
            padding: Padding.
            norm_layer: Norm layer.
            activation: Activation.
            stride: Stride.
            downsample: Downsample module that maps identity to correct dimensions. Default is an optionally strided
                1x1 Conv2d with BatchNorm2d, as per He et al. (2015) (`3.3. Network Architectures`, `Residual Network`,
                "option (B)").
            **kwargs: Keyword arguments for Conv2d layers.
        """
        Conv = get_nd_conv(nd)
        # print(norm_layer)
        # assert 1 == 0
        #         Norm = lookup_nn(norm_layer, nd=nd, call=False)
        print(norm_layer)
        if 'batch' in str(norm_layer).lower():
            Norm = lookup_nn(nn.BatchNorm2d, nd=nd, call=False)
        elif 'group' in str(norm_layer).lower():
            Norm = NormProxy(nn.GroupNorm, num_groups=32)

        super().__init__(
            in_channels, out_channels,
            block=nn.Sequential(
                Conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, stride=stride,
                     **kwargs),
                Norm(out_channels),
                lookup_nn(activation),
                Conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, **kwargs),
                Norm(out_channels),
            ),
            activation=activation, stride=stride, downsample=downsample, nd=nd, norm_layer=norm_layer
        )


def get_nn(item: Union[str, 'nn.Module', Type['nn.Module']], src=None, nd=None, call_if_type=False):
    print(item)
    # assert 1 == 0
    # ret = lookup_nn(item, src=src, nd=nd, call=False)
    #     if 'ResBlock' in item:
    ret = lookup_nn(ResBlock, src=src, nd=nd, call=False)

    if call_if_type and type(ret) is type:
        ret = ret()
    return ret

class BottleneckBlock(_ResBlock):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            mid_channels=None,
            compression=4,
            base_channels=64,
            norm_layer='BatchNorm2d',
            activation='ReLU',
            stride=1,
            downsample=None,
            nd=2,
            **kwargs
    ) -> None:
        """Bottleneck Block.

        Typical Bottleneck Block with variable kernel size and an included mapping of the identity to correct
        dimensions.

        References:
            - https://doi.org/10.1109/CVPR.2016.90
            - https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch

        Notes:
            - Similar to ``torchvision.models.resnet.Bottleneck``, with different interface and defaults.
            - Consistent with standard signature ``in_channels, out_channels, kernel_size, ...``.
            - Stride handled in bottleneck.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size.
            padding: Padding.
            mid_channels:
            compression: Compression rate of the bottleneck. The default 4 compresses 256 channels to 64=256/4.
            base_channels: Minimum number of ``mid_channels``.
            norm_layer: Norm layer.
            activation: Activation.
            stride: Stride.
            downsample: Downsample module that maps identity to correct dimensions. Default is an optionally strided
                1x1 Conv2d with BatchNorm2d, as per He et al. (2015) (`3.3. Network Architectures`, `Residual Network`,
                "option (B)").
            **kwargs: Keyword arguments for Conv2d layers.
        """
        Conv = get_nd_conv(nd)
        #Norm = lookup_nn(norm_layer, nd=nd, call=False)
        if 'batch' in norm_layer.lower():
            Norm = lookup_nn(nn.BatchNorm2d, nd=nd, call=False)
        elif 'group' in norm_layer.lower():
            Norm = lookup_nn(nn.GroupNorm, nd=nd, call=False)
        mid_channels = mid_channels or np.max([base_channels, out_channels // compression, in_channels // compression])
        super().__init__(
            in_channels, out_channels,
            block=nn.Sequential(
                Conv(in_channels, mid_channels, kernel_size=1, padding=0, bias=False, **kwargs),
                Norm(mid_channels),
                lookup_nn(activation),

                Conv(mid_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False, stride=stride,
                     **kwargs),
                Norm(mid_channels),
                lookup_nn(activation),

                Conv(mid_channels, out_channels, kernel_size=1, padding=0, bias=False, **kwargs),
                Norm(out_channels)
            ),
            activation=activation, stride=stride, downsample=downsample
        )

class ReadOut(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            kernel_size=3,
            padding=1,
            activation='relu',
            norm='batchnorm2d',
            final_activation=None,
            dropout=0.1,
            channels_mid=None,
            stride=1,
            nd=2,
            attention=None,
    ):
        super().__init__()
        Conv = get_nd_conv(nd)
        Norm = lookup_nn(norm, nd=nd, call=False)
        Dropout = lookup_nn(nn.Dropout2d, nd=nd, call=False)
        self.channels_out = channels_out
        if channels_mid is None:
            channels_mid = channels_in

        self.attention = None
        if attention is not None:
            if isinstance(attention, dict):
                attention_kwargs, = list(attention.values())
                attention, = list(attention.keys())
            else:
                attention_kwargs = {}
            self.attention = lookup_nn(attention, nd=nd, call=False)(channels_in, **attention_kwargs)

        self.block = nn.Sequential(
            Conv(channels_in, channels_mid, kernel_size, padding=padding, stride=stride),
            Norm(channels_mid),
            lookup_nn(activation),
            Dropout(p=dropout) if dropout else nn.Identity(),
            Conv(channels_mid, channels_out, 1),
        )

        if final_activation is ...:
            self.activation = lookup_nn(activation)
        else:
            self.activation = lookup_nn(final_activation)

    def forward(self, x):
        if self.attention is not None:
            x = self.attention(x)
        out = self.block(x)
        return self.activation(out)

# backbone.ppm

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from torchvision.models.segmentation.deeplabv3 import ASPP


class Ppm(nn.Module):
    def __init__(
            self,
            in_channels, out_channels,
            scales: Union[list, tuple] = (1, 2, 3, 6),
            kernel_size=1,
            norm='BatchNorm2d',
            activation='relu',
            concatenate=True,
            nd=2,
            **kwargs
    ):
        """Pyramid Pooling Module.

        References:
            - https://ieeexplore.ieee.org/document/8100143

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels per pyramid scale.
            scales: Pyramid scales. Default: (1, 2, 3, 6).
            kernel_size: Kernel size.
            norm: Normalization.
            activation: Activation.
            concatenate: Whether to concatenate module inputs to pyramid pooling output before returning results.
            **kwargs: Keyword arguments for ``nn.Conv2d``.
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        self.concatenate = concatenate
        self.out_channels = out_channels * len(scales) + in_channels * int(concatenate)
        Conv = get_nd_conv(nd)
        AdaptiveAvgPool = lookup_nn(nn.AdaptiveAvgPool2d, call=False, nd=nd)
        # norm = lookup_nn(norm, call=False, nd=nd)
        if 'batch' in norm.lower():
            norm = lookup_nn(nn.BatchNorm2d, call=False, nd=nd)
        elif 'group' in norm.lower():
            norm = lookup_nn(nn.GroupNorm, call=False, nd=nd)

        if 'leaky' in activation.lower():
            activation = lookup_nn(nn.LeakyReLU, call=False, nd=nd)
        elif 'relu' in activation.lower():
            activation = lookup_nn(nn.ReLU, call=False, nd=nd)

        for scale in scales:
            self.blocks.append(nn.Sequential(
                AdaptiveAvgPool(output_size=scale),
                Conv(in_channels, out_channels, kernel_size, **kwargs),
                norm(out_channels),
                activation(),
            ))

    def forward(self, x):
        # print(x.shape)
        prefix = [x] if self.concatenate else []

        # print(self.blocks)
        # tmp = []
        # print("#"*20)
        # for m in self.blocks:
        #     print(x.shape)
        #     print(m)
        #     tx = m(x)
        #     print(tx.shape)

        return torch.cat(prefix + [
            F.interpolate(m(x), x.shape[2:], mode='bilinear', align_corners=False) for m in self.blocks
        ], 1)

def append_pyramid_pooling_(module: nn.Sequential, out_channels, scales=(1, 2, 3, 6), method='ppm', in_channels=None,
                            **kwargs):
    if in_channels is None:
        in_channels = module.out_channels[-1]
    method = method.lower()
    if method == 'ppm':
        assert (out_channels % len(scales)) == 0
        p = Ppm(in_channels, out_channels, scales=scales, **kwargs)
        out_channels = p.out_channels
    elif method == 'aspp':
        scales = sorted(tuple(set(scales) - {1}))
        nd = kwargs.pop('nd', 2)
        assert nd == 2, NotImplementedError('Only nd=2 supported.')
        p = ASPP(in_channels, scales, out_channels, **kwargs)
    else:
        raise ValueError
    module.append(p)
    if hasattr(module, 'out_channels'):
        module.out_channels += (out_channels,)
    if hasattr(module, 'out_strides'):
        module.out_strides += module.out_strides[-1:]

# backbone.resnet
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet as tvr
from os.path import isfile
# from ..util.util import Dict, lookup_nn, get_nd_conv, get_nn, resolve_pretrained
from torch.hub import load_state_dict_from_url
# from .ppm import append_pyramid_pooling_
from typing import Type, Union, Optional
from pytorch_lightning.core.mixins import HyperparametersMixin

default_model_urls = {
    'ResNet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',  # IMAGENET1K_V1
    'ResNet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',  # IMAGENET1K_V1
    'ResNet50': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth',  # IMAGENET1K_V2
    'ResNet101': 'https://download.pytorch.org/models/resnet101-cd907fc2.pth',  # IMAGENET1K_V2
    'ResNet152': 'https://download.pytorch.org/models/resnet152-f82ba261.pth',  # IMAGENET1K_V2
    'ResNeXt50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth',  # IMAGENET1K_V2
    'ResNeXt101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',  # IMAGENET1K_V2
    'WideResNet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth',  # IMAGENET1K_V2
    'WideResNet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth',  # IMAGENET1K_V2
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, kernel_size=3,
            nd=2) -> nn.Conv2d:
    """3x3 convolution with padding"""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * nd
    if isinstance(dilation, int):
        dilation = (dilation,) * nd

    # Calculate padding for 'same' padding
    padding = tuple((ks - 1) * dil // 2 for ks, dil in zip(kernel_size, dilation))

    return get_nd_conv(nd)(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1, nd=2) -> nn.Conv2d:
    """1x1 convolution"""
    return get_nd_conv(nd)(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = tvr.BasicBlock.expansion
    forward = tvr.BasicBlock.forward

    def __init__(  # Port from torchvision (to support 3d and add more features)
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer='batchnorm2d',
            kernel_size=3,
            nd=2
            # nn.GroupNorm
    ) -> None:
        super().__init__()
        norm_layer = lookup_nn(norm_layer, call=False, nd=nd)
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride, nd=nd, kernel_size=kernel_size)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, nd=nd)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

class Bottleneck(nn.Module):
    expansion: int = tvr.Bottleneck.expansion
    forward = tvr.Bottleneck.forward

    def __init__(  # Port from torchvision (to support 3d and add more features)
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer='batchnorm2d',
            kernel_size=3,
            nd=2
    ) -> None:
        super().__init__()
        norm_layer = lookup_nn(norm_layer, call=False, nd=nd)
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width, nd=nd)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, kernel_size=kernel_size, nd=nd)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, nd=nd)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

def _make_layer(  # Port from torchvision (to support 3d)
        d,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        kernel_size: int = 3,
        nd=2,
        secondary_block=None,
        downsample_method=None,
) -> nn.Sequential:
    """

    References:
        - [1] https://arxiv.org/abs/1812.01187.pdf

    Args:
        self:
        block:
        planes:
        blocks:
        stride:
        dilate:
        kernel_size:
        nd:
        secondary_block:
        downsample_method: Downsample method. None: 1x1Conv with stride, Norm (standard ResNet),
            'avg': AvgPool, 1x1Conv, Norm (ResNet-D in [1])

    Returns:

    """
    if secondary_block is not None:
        secondary_block = get_nn(secondary_block, nd=nd)
    norm_layer = d['norm_layer']
    downsample = None
    previous_dilation = d['dilation']
    if dilate:
        d['dilation'] *= stride
        stride = 1
    if stride != 1 or d['inplanes'] != planes * block.expansion:
        if downsample_method is None or stride <= 1:
            downsample = nn.Sequential(
                conv1x1(d['inplanes'], planes * block.expansion, stride, nd=nd),
                norm_layer(planes * block.expansion),
            )
        elif downsample_method == 'avg':
            downsample = nn.Sequential(
                get_nn(nn.AvgPool2d, nd=nd)(2, stride=stride),
                conv1x1(d['inplanes'], planes * block.expansion, nd=nd),
                norm_layer(planes * block.expansion),
            )
        else:
            raise ValueError(f'Unknown downsample_method: {downsample_method}')

    layers = []
    layers.append(
        block(d['inplanes'], planes, stride, downsample, d['groups'], d['base_width'], previous_dilation, norm_layer,
              kernel_size=kernel_size, nd=nd))
    d['inplanes'] = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(
            d['inplanes'],
            planes,
            groups=d['groups'],
            base_width=d['base_width'],
            dilation=d['dilation'],
            norm_layer=norm_layer,
            kernel_size=kernel_size,
            nd=nd,
        ))
    if secondary_block is not None:
        layers.append(secondary_block(d.inplanes, nd=nd))  # must be preconfigured and not change channels
    return nn.Sequential(*layers)

def make_res_layer(block, inplanes, planes, blocks, norm_layer=nn.BatchNorm2d, base_width=64, groups=1, stride=1,
                   dilation=1, dilate=False, nd=2, secondary_block=None, downsample_method=None, kernel_size=3,
                   **kwargs) -> nn.Module:
    """

    Args:
        block: Module class. For example `BasicBlock` or `Bottleneck`.
        inplanes: Number of in planes
        planes: Number of planes
        blocks: Number of blocks
        norm_layer: Norm Module class
        base_width: Base width. Acts as a factor of the bottleneck size of the Bottleneck block and is used with groups.
        groups:
        stride:
        dilation:
        dilate:
        nd:
        secondary_block:
        downsample_method:
        kernel_size:
        kwargs:

    Returns:

    """
    norm_layer = lookup_nn(norm_layer, nd=nd, call=False)
#     d = dict(inplanes=inplanes, _norm_layer=norm_layer, base_width=base_width,
#              groups=groups, dilation=dilation)  # almost a ResNet
    d = {
        'inplanes': inplanes,
        'norm_layer': norm_layer,
        'base_width': base_width,
        'groups': groups,
        'dilation': dilation
    }

    return _make_layer(d=d, block=block, planes=planes, blocks=blocks, stride=stride, dilate=dilate, nd=nd,
                       secondary_block=secondary_block, downsample_method=downsample_method, kernel_size=kernel_size)

def _apply_mapping_rules(key, rules: dict):
    for prefix, repl in rules.items():
        if key.startswith(prefix):
            key = key.replace(prefix, repl, 1)
    return key

def map_state_dict(in_channels, state_dict, fused_initial):
    """Map state dict.

    Map state dict from torchvision format to celldetection format.

    Args:
        in_channels: Number of input channels.
        state_dict: State dict.
        fused_initial:

    Returns:
        State dict in celldetection format.
    """
    mapping = {}
    for k, v in state_dict.items():
        if 'fc' in k:  # skip fc
            continue
        if k.startswith('conv1.') and v.data.shape[1] != in_channels:  # initial layer, img channels might differ
            v.data = F.interpolate(v.data[None], (in_channels,) + v.data.shape[-2:]).squeeze(0)
        if fused_initial:
            rules = {'conv1.': '0.0.', 'bn1.': '0.1.', 'layer1.': '0.4.', 'layer2.': '1.', 'layer3.': '2.',
                     'layer4.': '3.', 'layer5.': '4.'}
        else:
            rules = {'conv1.': '0.0.', 'bn1.': '0.1.', 'layer1.': '1.1.', 'layer2.': '2.', 'layer3.': '3.',
                     'layer4.': '4.', 'layer5.': '5.'}
        mapping[_apply_mapping_rules(k, rules)] = v
    return mapping

class ResNet(nn.Sequential, HyperparametersMixin):
    def __init__(self, in_channels, *body: nn.Module, initial_strides=2, base_channel=64, initial_pooling=True,
                 final_layer=None, final_activation=None, fused_initial=True, pretrained=False,
                 pyramid_pooling=False, pyramid_pooling_channels=64, pyramid_pooling_kwargs=None, nd=2, **kwargs):
        assert len(body) > 0
        body = list(body)
        Conv = get_nd_conv(nd)
        Norm = lookup_nn(nn.BatchNorm2d, nd=nd, call=False)
        MaxPool = lookup_nn(nn.MaxPool2d, nd=nd, call=False)
        initial = [
            Conv(in_channels, base_channel, 7, padding=3, bias=False, stride=initial_strides),
            Norm(base_channel),
            nn.ReLU(inplace=True)
        ]
        pool = MaxPool(kernel_size=3, stride=2, padding=1) if initial_pooling else nn.Identity()
        if fused_initial:
            initial += [pool, body[0]]
        else:
            body[0] = nn.Sequential(pool, body[0])
        initial = nn.Sequential(*initial)
        components = [initial] + list(body[1:] if fused_initial else body)
        if final_layer is not None:
            components += [final_layer]
        if final_activation is not None:
            components += [lookup_nn(final_activation)]
        super(ResNet, self).__init__(*components)
        if pretrained:
            state_dict = resolve_pretrained(pretrained, in_channels=in_channels, fused_initial=fused_initial,
                                            state_dict_mapper=map_state_dict)
            self.load_state_dict(state_dict, strict=kwargs.get('pretrained_strict', True))
        if pyramid_pooling:
            pyramid_pooling_kwargs = {} if pyramid_pooling_kwargs is None else pyramid_pooling_kwargs
            append_pyramid_pooling_(self, pyramid_pooling_channels, nd=nd, **pyramid_pooling_kwargs)

class BottleResNet(ResNet):
    def __init__(self, in_channels, out_channels=0, layers=(3, 4, 6, 3), base_channel=64, fused_initial=True,
                 kernel_size=3, per_layer_kernel_sizes: dict = None, nd=2, **kwargs):
        if per_layer_kernel_sizes is None:
            per_layer_kernel_sizes = {}
        if isinstance(per_layer_kernel_sizes, (tuple, list)):
            per_layer_kernel_sizes = {i: v for i, v in enumerate(per_layer_kernel_sizes)}
        self.save_hyperparameters()
        ex = Bottleneck.expansion
        self.out_channels = oc = (base_channel * 4, base_channel * 8, base_channel * 16, base_channel * 32)
        self.out_strides = (4, 8, 16, 32)
        if out_channels and 'final_layer' not in kwargs.keys():
            kwargs['final_layer'] = nn.Conv2d(self.out_channels[-1], out_channels, 1)
        super(BottleResNet, self).__init__(
            in_channels,
            make_res_layer(Bottleneck, base_channel, oc[0] // ex, layers[0], stride=1, nd=nd,
                           kernel_size=per_layer_kernel_sizes.get(0, kernel_size), **kwargs),
            make_res_layer(Bottleneck, base_channel * 4, oc[1] // ex, layers[1], stride=2, nd=nd,
                           kernel_size=per_layer_kernel_sizes.get(1, kernel_size), **kwargs),
            make_res_layer(Bottleneck, base_channel * 8, oc[2] // ex, layers[2], stride=2, nd=nd,
                           kernel_size=per_layer_kernel_sizes.get(2, kernel_size), **kwargs),
            make_res_layer(Bottleneck, base_channel * 16, oc[3] // ex, layers[3], stride=2, nd=nd,
                           kernel_size=per_layer_kernel_sizes.get(3, kernel_size), **kwargs),
            base_channel=base_channel, fused_initial=fused_initial, nd=nd, **kwargs
        )
        if not fused_initial:
            self.out_channels = (base_channel,) + self.out_channels
            self.out_strides = (2,) + self.out_strides


class ResNet101(BottleResNet):
    def __init__(self, in_channels, out_channels=0, pretrained=False, nd=2, **kwargs):
        if pretrained is True and nd == 2:
            pretrained = default_model_urls['ResNet101']
        super(ResNet101, self).__init__(in_channels, out_channels=out_channels, layers=(3, 4, 23, 3),
                                        pretrained=pretrained, nd=nd, **kwargs)
        self.hparams.clear()
        self.save_hyperparameters()

class ResNeXt101_32x8d(BottleResNet):
    def __init__(self, in_channels, out_channels=0, pretrained=False, nd=2, **kwargs):
        if pretrained is True and nd == 2:
            pretrained = default_model_urls['ResNeXt101_32x8d']
        super(ResNeXt101_32x8d, self).__init__(in_channels, out_channels=out_channels, layers=(3, 4, 23, 3), groups=32,
                                               base_width=8, pretrained=pretrained, nd=nd, **kwargs)
        self.hparams.clear()
        self.save_hyperparameters()

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, ExtraFPNBlock
from collections import OrderedDict
from typing import List, Dict, Type, Union, Tuple
from functools import partial
import numpy as np
from pytorch_lightning.core.mixins import HyperparametersMixin
class UNetEncoder(nn.Sequential):
    def __init__(self, in_channels, depth=5, base_channels=64, factor=2, pool=True, block_cls: Type[nn.Module] = None,
                 nd=2):
        """U-Net Encoder.

        Args:
            in_channels: Input channels.
            depth: Model depth.
            base_channels: Base channels.
            factor: Growth factor of base_channels.
            pool: Whether to use max pooling or stride 2 for downsampling.
            block_cls: Block class. Callable as `block_cls(in_channels, out_channels, stride=stride)`.
        """
        if block_cls is None:
            block_cls = partial(TwoConvNormRelu, nd=nd)
        else:
            block_cls = get_nn(block_cls, nd=nd)
        MaxPool = get_nd_max_pool(nd)
        layers = []
        self.out_channels = []
        self.out_strides = list(range(1, depth + 1))
        for i in range(depth):
            in_c = base_channels * int(factor ** (i - 1)) * int(i > 0) + int(i <= 0) * in_channels
            out_c = base_channels * (factor ** i)
            self.out_channels.append(out_c)
            block = block_cls(in_c, out_c, stride=int((not pool and i > 0) + 1))
            if i > 0 and pool:
                block = nn.Sequential(MaxPool(2, stride=2), block)
            layers.append(block)
        super().__init__(*layers)

class GeneralizedUNet(FeaturePyramidNetwork):
    def __init__(
            self,
            in_channels_list,
            out_channels: int,
            block_cls: nn.Module,
            block_kwargs: dict = None,
            final_activation=None,
            interpolate='nearest',
            final_interpolate=None,
            initialize=True,
            keep_features=True,
            bridge_strides=True,
            bridge_block_cls: 'nn.Module' = None,
            bridge_block_kwargs: dict = None,
            secondary_block: 'nn.Module' = None,
            in_strides_list: Union[List[int], Tuple[int]] = None,
            out_channels_list: Union[List[int], Tuple[int]] = None,
            nd=2,
            **kwargs
    ):
        super().__init__([], 0, extra_blocks=kwargs.get('extra_blocks'))
        block_kwargs = {} if block_kwargs is None else block_kwargs
        Conv = get_nd_conv(nd)
        if out_channels_list is None:
            out_channels_list = in_channels_list
        if in_strides_list is None or bridge_strides is False:  # if not provided, act as if it is starting at stride 1
            in_strides_list = [2 ** i for i in range(len(in_channels_list))]

        # Optionally bridge stride gaps
        self.bridges = np.log2(in_strides_list[0])
        assert self.bridges % 1 == 0
        self.bridges = int(self.bridges)
        if bridge_block_cls is None:
            bridge_block_cls = partial(TwoConvNormRelu, bias=False)
        else:
            bridge_block_cls = get_nn(bridge_block_cls, nd=nd)
        bridge_block_kwargs = {} if bridge_block_kwargs is None else bridge_block_kwargs
        update_dict_(bridge_block_kwargs, block_kwargs, ('activation', 'norm_layer'))
        if self.bridges:
            num = len(in_channels_list)
            for _ in range(self.bridges):
                in_channels_list = (0,) + tuple(in_channels_list)
                if len(out_channels_list) < num + self.bridges - 1:
                    # out_channels_list = (2 ** int(np.log2(out_channels_list[0]) - 1e-8),) + tuple(out_channels_list)
                    out_channels_list = (out_channels_list[0],) + tuple(out_channels_list)

        # Build decoder
        self.cat_order = kwargs.get('cat_order', 0)
        assert self.cat_order in (0, 1)
        self.block_channel_reduction = kwargs.get('block_channel_reduction', False)  # whether block reduces in_channels
        self.block_interpolate = kwargs.get('block_interpolate', False)  # whether block handles interpolation
        self.block_cat = kwargs.get('block_cat', False)  # whether block handles cat
        self.bridge_block_interpolate = kwargs.get('bridge_block_interpolate', False)  # whether block handles interpol.
        self.apply_cat = {}
        self.has_lat = {}
        len_in_channels_list = len(in_channels_list)
        for i in range(len_in_channels_list):
            # Inner conv
            if i > 0:
                inner_ouc = out_channels_list[i - 1]
                inner_inc = out_channels_list[i] if i < len_in_channels_list - 1 else in_channels_list[i]
                if not self.block_channel_reduction and inner_inc > 0 and inner_ouc < inner_inc:
                    inner = Conv(inner_inc, inner_ouc, 1)
                else:
                    inner = nn.Identity()
                self.inner_blocks.append(inner)

            if i < len_in_channels_list - 1:
                # Layer block channels
                lat = in_channels_list[i]
                if self.block_channel_reduction:
                    inc = out_channels_list[i + 1] if i < len_in_channels_list - 2 else in_channels_list[i + 1]
                else:
                    inc = min(out_channels_list[i:i + 2])
                ouc = out_channels_list[i]

                # Build decoder block
                self.apply_cat[i] = False
                self.has_lat[i] = has_lat = lat > 0
                cls, kw = block_cls, block_kwargs
                if not has_lat:  # bridge block
                    self.has_lat[i] = False
                    cls, kw = bridge_block_cls, bridge_block_kwargs
                    inp = inc,
                elif self.block_cat:  # block_cls handles merging
                    inp = inc, lat
                else:  # normal cat
                    self.apply_cat[i] = True
                    inp = inc + lat,
                layer_block = cls(*inp, ouc, nd=nd, **kw)
                if secondary_block is not None:  # must be preconfigured and not change channels
                    layer_block = nn.Sequential(layer_block, secondary_block(ouc, nd=nd))
                self.layer_blocks.append(layer_block)

        self.depth = len(self.layer_blocks)
        self.interpolate = interpolate

        self.keep_features = keep_features
        self.features_prefix = 'encoder'
        self.out_layer = Conv(out_channels_list[0], out_channels, 1) if out_channels > 0 else None
        self.nd = nd
        self.final_interpolate = final_interpolate
        if self.final_interpolate is None:
            self.final_interpolate = get_nd_linear(nd)
        self.final_activation = None if final_activation is None else lookup_nn(final_activation)
        self.out_channels_list = out_channels_list
        self.out_channels = out_channels if out_channels else out_channels_list

        if initialize:
            for m in self.modules():
                if isinstance(m, Conv):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: Dict[str, Tensor], size: List[int], guild=False) -> Union[Dict[str, Tensor], Tensor]:
        """

        Args:
            x: Input dictionary. E.g. {
                    0: Tensor[1, 64, 128, 128]
                    1: Tensor[1, 128, 64, 64]
                    2: Tensor[1, 256, 32, 32]
                    3: Tensor[1, 512, 16, 16]
                }
            size: Desired final output size. If set to None output remains as it is.

        Returns:
            Output dictionary. For each key in `x` a corresponding output is returned; the final output
            has the key `'out'`.
            E.g. {
                out: Tensor[1, 2, 128, 128]
                0: Tensor[1, 64, 128, 128]
                1: Tensor[1, 128, 64, 64]
                2: Tensor[1, 256, 32, 32]
                3: Tensor[1, 512, 16, 16]
            }
        """
        features = x
        names = list(x.keys())
        x = list(x.values())

        last_inner = x[-1]

        results = [last_inner]
        kw = {} if self.interpolate == 'nearest' else {'align_corners': False}
        # from down(low resolution) to top(high resolution)
        for i in range(self.depth - 1, -1, -1):
            # print(i) # 4, 3, 2, 1, 0
            lateral = lateral_size = None
            if self.has_lat[i]: # yes
                # print('lat...')
                lateral = x[i - self.bridges]
                lateral_size = lateral.shape[2:]

            inner_top_down = last_inner

            # print(lateral.shape, inner_top_down.shape)
            if self.interpolate and ((not self.block_interpolate and lateral is not None) or (
                    not self.bridge_block_interpolate and lateral is None)): # yes
                inner_top_down = F.interpolate(  # TODO: scale factor entails shape assumption
                    inner_top_down, **(dict(scale_factor=2) if lateral_size is None else dict(size=lateral_size)),
                    mode=self.interpolate, **kw)
                # print("interp.")

            inner_top_down = self.get_result_from_inner_blocks(inner_top_down, i)
            if self.apply_cat[i]: # yes
                # print('concat..')
                if self.cat_order == 0:
                    cat = lateral, inner_top_down
                else:
                    cat = inner_top_down, lateral
                layer_block_inputs = torch.cat(cat, 1)

            elif lateral is None:
                layer_block_inputs = inner_top_down
            else:
                layer_block_inputs = inner_top_down, lateral

            # print(lateral.shape, inner_top_down.shape)
            last_inner = self.get_result_from_layer_blocks(layer_block_inputs, i)
            results.insert(0, last_inner)

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        if size is None:
            final = results[0]
        else:
            final = F.interpolate(last_inner, size=size, mode=self.final_interpolate, align_corners=False)
        if self.out_layer is not None:
            final = self.out_layer(final)
        if self.final_activation is not None:
            final = self.final_activation(final)
        if self.out_layer is not None:
            return final
        results.insert(0, final)
        names.insert(0, 'out')
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        # print("inner unet")
        # for k, v in out.items(): # 0, 1, 2, 3, 4, 5
        #     print(k, v.shape)

        # assert 1 == 0
        if self.keep_features and not guild:
            out.update(OrderedDict([('.'.join([self.features_prefix, k]), v) for k, v in features.items()]))

        # for k, v in out.items(): # 0, 1, 2, 3, 4, 5, encoder.0,1,2,3,4,5
        #     print(k, v.shape)
        #
        # assert 1 == 0

        return out

class BackboneAsUNet(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, block, block_kwargs: dict = None,
                 final_activation=None, interpolate='nearest', ilg=None, nd=2, in_strides_list=None, **kwargs):
        super(BackboneAsUNet, self).__init__()
        if ilg is None:
            ilg = isinstance(backbone, nn.Sequential)
        if block is None:
            block = TwoConvNormRelu  # it's called with nd
        else:
            block = get_nn(block, nd=nd)
        self.nd = nd
        pretrained_cfg = backbone.__dict__.get('pretrained_cfg', {})

        self.use_guild = kwargs.get("use_guild", None)
        # print(self.use_guild)
        # assert 1 == 0
        # change this, we can normalize the input before input to model
        # if kwargs.pop('normalize', True):
        #     self.normalize = Normalize(mean=kwargs.get('inputs_mean', pretrained_cfg.get('mean', 0.)),
        #                                std=kwargs.get('inputs_std', pretrained_cfg.get('std', 1.)),
        #                                assert_range=kwargs.get('assert_range', (0., 1.)))
        # else:
        #     self.normalize = None
        #
        # if self.use_guild:
        self.normalize = None
            # assert self.normalize is None

        # assert self.normalize is not None

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers) if ilg else backbone

        self.intermediate_blocks = kwargs.get('intermediate_blocks')
        if self.intermediate_blocks is not None:
            in_channels_list = in_channels_list + type(in_channels_list)(self.intermediate_blocks.out_channels)
            if in_strides_list is not None:
                in_strides_list = in_strides_list + type(in_strides_list)(
                    [i * in_strides_list[-1] for i in self.intermediate_blocks.out_strides])

        self.unet = GeneralizedUNet(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            block_cls=block,
            block_kwargs=block_kwargs,
            # extra_blocks=LastLevelMaxPool(),
            final_activation=final_activation,
            interpolate=interpolate,
            in_strides_list=in_strides_list,
            nd=nd,
            **kwargs
        )

        fusion_text_vis = nn.ModuleDict()
        for i in range(len(in_channels_list)):
            # print(i, in_channels_list[i])
            fusion_text_vis[str(i)] = nn.Sequential(
                CoordConv2(in_channels_list[i] + 768, in_channels_list[i]),
                nn.Conv2d(in_channels_list[i], in_channels_list[i], 3, 1, bias=False),
                nn.BatchNorm2d(in_channels_list[i]),
                nn.ReLU(inplace=True),
                # nn.Conv2d(in_channels_list[i], in_channels_list[i], 3, 1, bias=False),

            )
            # print(fusion_text_vis[str(i)])

        self.text_vis_fusion = fusion_text_vis

        self.fusion_type = kwargs.get("fusion_type", 'fusion')



        # print(self.fusion_text)
        # assert 1 == 0
        self.out_channels = list(self.unet.out_channels_list)  # list(in_channels_list)
        # self.out_strides = kwargs.get('in_stride_list')
        self.nd = nd

    def forward(self, inputs, text_features=None):
        # print("in backbone: unet")
        x = inputs
        # print(x.shape)
        if self.normalize is not None:
            raise
            x = self.normalize(x)

        x = self.body(x)  # encoder results

        tx = {}
        for k in x.keys():
            xi = x[k]
            ch, cw = xi.shape[-2: ]
            text_featuresi = text_features.unsqueeze(-1).unsqueeze(-1) # [B, 768, 1, 1]
            text_featuresi = text_featuresi.repeat(1, 1, ch, cw)
            xi_text = torch.cat([xi, text_featuresi], dim=1)
            xi_text = self.text_vis_fusion[k](xi_text)
            tx[k] = xi_text
        x = tx

        # for k, v in x.items(): # 0, 1, 2, 3, 4, 5
        #     print(k, v.shape)

        # assert 1 == 0
        if self.intermediate_blocks is not None:
            x = self.intermediate_blocks(x)
        # print(type(x))
        # for k, v in x.items(): # 0, 1, 2, 3, 4, 5
        #     print(k, v.shape)


        x_ori = self.unet(x, size=inputs.shape[-self.nd:])

        # print(type(x)) # <class 'collections.OrderedDict'>
        # for k, v in x.items(): # 0, 1, 2, 3, 4, 5, encoder.0, 1, 2, 3, 4, 5
        #     print(k, v.shape)
        if self.use_guild == 'seg' and self.fusion_type!='dattn':
            # assert guild is not None
            # print(guild.shape)
            x_guild = self.guild_net(x, size=inputs.shape[-self.nd:], guild=True) # 0, 1, 2, 3, 4, 5, encoder.0, 1, 2, 3, 4, 5
        else:
            x_guild = None

        # return x_ori
        # assert 1 == 0
        return x_ori

class ExtraUNetBlock(ExtraFPNBlock):
    def __init__(self, out_channels: Tuple[int], out_strides: Tuple[int]):
        super().__init__()
        self.out_channels = out_channels
        self.out_strides = out_strides

    def forward(
            self,
            results: List[Tensor],
            x: List[Tensor],
            names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        pass



class IntermediateUNetBlock(nn.Module):
    def __init__(self, out_channels: Tuple[int], out_strides: Tuple[int]):
        super().__init__()
        self.out_channels = out_channels
        self.out_strides = out_strides

    def forward(
            self,
            x: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        pass

class ResNet50(BottleResNet):
    def __init__(self, in_channels, out_channels=0, pretrained=False, nd=2, **kwargs):
        if pretrained is True and nd == 2:
            pretrained = default_model_urls['ResNet50']
        super(ResNet50, self).__init__(in_channels, out_channels=out_channels, layers=(3, 4, 6, 3),
                                       pretrained=pretrained, nd=nd, **kwargs)
        self.hparams.clear()
        self.save_hyperparameters()


class UNet(BackboneAsUNet, HyperparametersMixin):
    def __init__(self, backbone, out_channels: int, return_layers: dict = None,
                 block: Type[nn.Module] = None, block_kwargs: dict = None, final_activation=None,
                 interpolate='nearest', nd=2, **kwargs):
        """U-Net.

        References:
            - https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

        Args:
            backbone: Backbone instance.
            out_channels: Output channels. If set to ``0``, the output layer is omitted.
            return_layers: Return layers used to extract layer features from backbone.
                Dictionary like `{backbone_layer_name: out_name}`.
                Note that this influences how outputs are computed, as the input for the upsampling
                is gathered by `IntermediateLayerGetter` based on given dict keys.
            block: Main block. Default: ``TwoConvNormRelu``.
            block_kwargs: Block keyword arguments.
            final_activation: Final activation function.
            interpolate: Interpolation.
            nd: Spatial dimensions.
            **kwargs: Additional keyword arguments.
        """
        if block is None:
            block = partial(TwoConvNormRelu, nd=nd)
        else:
            block = get_nn(block, nd=nd)
        names = [name for name, _ in backbone.named_children()]  # assuming ordered
        if return_layers is None:
            return_layers = {n: str(i) for i, n in enumerate(names)}
        layers = {str(k): (str(names[v]) if isinstance(v, int) else str(v)) for k, v in return_layers.items()}
        in_channels_list = list(backbone.out_channels)
        in_strides_list = backbone.__dict__.get('out_strides')
        extra_blocks = kwargs.get('extra_blocks')
        if extra_blocks is not None:
            in_channels_list = in_channels_list + type(in_channels_list)(extra_blocks.out_channels)
            if in_strides_list is not None:
                in_strides_list = in_strides_list + type(in_strides_list)(
                    [i * in_strides_list[-1] for i in extra_blocks.out_strides])
        super().__init__(
            backbone=backbone,
            return_layers=layers,
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            block=block,
            block_kwargs=block_kwargs,
            final_activation=final_activation if out_channels else None,
            interpolate=interpolate,
            nd=nd,
            in_strides_list=in_strides_list,
            **kwargs
        )


def _ni_pretrained(pretrained):
    if pretrained:
        raise NotImplementedError('The `pretrained` option is not yet available for this model.')


def _default_unet_kwargs(backbone_kwargs, pretrained=False):
    _ni_pretrained(pretrained)
    kw = dict()
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw

class ResUNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        """Residual U-Net.

        U-Net with residual blocks.

        References:
            - https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``ResBlock``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        self.save_hyperparameters()
        if block_cls is None:
            block_cls = partial(ResBlock, nd=nd)
        else:
            block_cls = get_nn(block_cls, nd=nd)
        super().__init__(
            UNetEncoder(in_channels=in_channels, block_cls=block_cls, nd=nd,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs
        )
def _default_res_kwargs(backbone_kwargs, pretrained=False):
    kw = dict(fused_initial=False, pretrained=pretrained)
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw

class ResNet50UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ResNet50(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)


class ResNet101UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ResNet101(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

    # __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNet 101')

class ResNeXt101UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(
            ResNeXt101_32x8d(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
            out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)




from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
# from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import WeightsEnum
# from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
# import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth

import copy
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig(_MBConvConfig):
    # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    def __init__(
            self,
            expand_ratio: float,
            kernel: int,
            stride: int,
            input_channels: int,
            out_channels: int,
            num_layers: int,
            width_mult: float = 1.0,
            depth_mult: float = 1.0,
            block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class FusedMBConvConfig(_MBConvConfig):
    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
            self,
            expand_ratio: float,
            kernel: int,
            stride: int,
            input_channels: int,
            out_channels: int,
            num_layers: int,
            block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = FusedMBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)


class MBConv(nn.Module):
    def __init__(
            self,
            cnf: MBConvConfig,
            stochastic_depth_prob: float,
            norm_layer: Callable[..., nn.Module],
            se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class FusedMBConv(nn.Module):
    def __init__(
            self,
            cnf: FusedMBConvConfig,
            stochastic_depth_prob: float,
            norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            # project
            layers.append(
                Conv2dNormActivation(
                    expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
                )
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

class EfficientNet(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
            dropout: float,
            stochastic_depth_prob: float = 0.2,
            num_classes: int = 1000,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            last_channel: Optional[int] = None,
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()
        _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
                isinstance(inverted_residual_setting, Sequence)
                and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.out_channels = [24, 24, 48, 64, 128, 160, 256, 1280]

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # print(len(self.features))
        # print(self.features)
        # x = self.features(x)
        res = {}
        for idx, e in enumerate(self.features):
            x = e(x)
            # print(x.shape)
            res[str(idx)] = x

        return res
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        #
        # x = self.classifier(x)

        # return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)



def _efficientnet(
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        last_channel: Optional[int],
        weights: Optional[WeightsEnum],
        progress: bool,
        **kwargs: Any,
) -> EfficientNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = EfficientNet(inverted_residual_setting, dropout, last_channel=last_channel, **kwargs)
    model.load_state_dict(torch.load('./efficientnet_v2_s-dd5fe13b.pth', map_location=torch.device('cpu')), strict=True)
    del model.avgpool
    del model.classifier

    print("loaded pretrained weight...")
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


def _efficientnet_conf(
        arch: str,
        **kwargs: Any,
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]
    if arch.startswith("efficientnet_b"):
        bneck_conf = partial(MBConvConfig, width_mult=kwargs.pop("width_mult"), depth_mult=kwargs.pop("depth_mult"))
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None
    elif arch.startswith("efficientnet_v2_s"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_m"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 3),
            FusedMBConvConfig(4, 3, 2, 24, 48, 5),
            FusedMBConvConfig(4, 3, 2, 48, 80, 5),
            MBConvConfig(4, 3, 2, 80, 160, 7),
            MBConvConfig(6, 3, 1, 160, 176, 14),
            MBConvConfig(6, 3, 2, 176, 304, 18),
            MBConvConfig(6, 3, 1, 304, 512, 5),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_l"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 32, 4),
            FusedMBConvConfig(4, 3, 2, 32, 64, 7),
            FusedMBConvConfig(4, 3, 2, 64, 96, 7),
            MBConvConfig(4, 3, 2, 96, 192, 10),
            MBConvConfig(6, 3, 1, 192, 224, 19),
            MBConvConfig(6, 3, 2, 224, 384, 25),
            MBConvConfig(6, 3, 1, 384, 640, 7),
        ]
        last_channel = 1280
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel



def efficientnet_v2_s(
        *, weights=None, progress: bool = True, **kwargs: Any
) -> EfficientNet:

    weights = None
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.2),
        last_channel,
        weights,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        **kwargs,
    )

class EfficientUNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(efficientnet_v2_s(),
                         out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

# our spine model
# -*- coding: utf-8 -*-
import math

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from typing import Dict, List, Optional, Tuple
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Sequential as Seq
from torchvision import transforms as trans
import cv2
from scipy.optimize import linear_sum_assignment
import clip

random_seed(42)


class SpineModel(torch.nn.Module):

    #     @configurable
    def __init__(self,
                 vis_period: int = 0,
                 ):
        super().__init__()

        self.num_classes = 3
        self.register_buffer("pixel_mean", torch.Tensor([103.530, 116.280, 123.675]).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor([1.0, 1.0, 1.0]).view(-1, 1, 1))

        self.vis_period = -1
        if vis_period > 0:
            # assert input_format is not None, "input_format is required for visualization!"
            assert 1 == 0

        self.time = 1

        # add model
        self.clip_encoder, _ = clip.load("ViT-L/14@336px")
        # freeze clip
        for key, value in self.clip_encoder.named_parameters(recurse=True):
            value.requires_grad = False
        # print(self.clip_encoder)
        # assert 1 == 0
        # kw = {'inputs_mean': [0.485, 0.456, 0.406], 'inputs_std': [0.229, 0.224, 0.225],
        #       'bridge_strides': False, 'block_cls': 'ResBlock',
        #       'block_kwargs': {'activation': 'LeakyReLU',
        #                        'norm_layer': {'NormProxy': {'norm': 'GroupNorm', 'num_groups': 32}}},
        #       'backbone_kwargs': {'pyramid_pooling': True, 'pyramid_pooling_channels': 64,
        #                           'pyramid_pooling_kwargs': {'method': 'ppm', 'concatenate': False}},
        #       'pretrained': True,
        #       'in_channels': 3,
        #       'classes': 3,
        #       'out_channels': 0,
        #       'interpolation': 'bilinear',
        #       }

        kw = {'inputs_mean': [0.485, 0.456, 0.406], 'inputs_std': [0.229, 0.224, 0.225],
              'bridge_strides': False, 'block_cls': 'ResBlock',
              'block_kwargs': {'activation': 'LeakyReLU',
                               # 'norm_layer': {'NormProxy': {'norm': 'GroupNorm', 'num_groups': 24}}
                               'norm_layer': {'NormProxy': {'norm': 'BatchNorm2d'}},
                               },
              'backbone_kwargs': {'pyramid_pooling': True, 'pyramid_pooling_channels': 64,
                                  'pyramid_pooling_kwargs': {'method': 'ppm', 'concatenate': False}},
              'pretrained': True,
              'in_channels': 3,
              'classes': 3,
              'out_channels': 0,
              'interpolation': 'bilinear',
              }

        # backbone
        # print(**kw)
        # self.backbone = ResNet101UNet(**kw)
        # self.backbone = ResNeXt101UNet(**kw)
        # self.backbone = ResNet50UNet(**kw)
        self.backbone = EfficientUNet(**kw)
        # d = torch.load('./efficientnet_v2_s-dd5fe13b.pth', map_location=torch.device('cpu'))
        # print(type(d))
        # print(d.keys())
        # print(type(self.backbone.state_dict()))
        # print(self.backbone.state_dict().keys())
        # assert 1 == 0
        # self.backbone.load_state_dict(torch.load('./efficientnet_v2_s-dd5fe13b.pth', map_location=torch.device('cpu')))
        # replace_bn_with_gn(self.backbone)
        #         replace_bn_with_syncbn(self.backbone)

        # for name, param in self.backbone.named_parameters():
        #     print(name)
        # self.backbone = ResNet101UNet(3, 0, backbone_kwargs=kw['backbone_kwargs'], pretrained=kw['pretrained'], block_cls=kw['backbone_kwargs']['block_cls'])
        # print(self.backbone)
        # assert 1 == 0
        # cls. head
        # take 4. layer
        num_of_groups = 100
        num_classes = 3
        num_features = 2048
        decoder_embedding = 768
        zsl = 0
        # self.cls_head = MLDecoder(num_classes=num_classes, initial_num_features=num_features, num_of_groups=num_of_groups,
        #                      decoder_embedding=decoder_embedding, zsl=zsl)

        self.mean_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.cls_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(128, num_classes),
        )
        # replace_bn_with_gn(self.cls_head)
        #         replace_bn_with_syncbn(self.cls_head)

        #         print(self.cls_head)
        # seg. head
        # take 0. layer
        self.feature_cls = '7'
        self.feature_seg = '0'

        self.seg_head = ReadOut(
            24, 1,
            kernel_size=kw.get('kernel_size_uncertainty', 3),
            padding=kw.get('kernel_size_uncertainty', 3) // 2,
            channels_mid=24,
            stride=1,
            final_activation='sigmoid',
            dropout=0.1,
            activation=kw.get('head_activation', 'relu')
        )
        # replace_bn_with_gn(self.seg_head)
        #         replace_bn_with_syncbn(self.seg_head)

        #         print(self.seg_head)
        # assert 1 == 0
        # multi-model fusion
        self.fusion_cls = nn.Sequential(
            nn.Conv2d(1280 + 768, 1280, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
            # nn.Conv2d(2048, 2048, kernel_size=3, padding=1, bias=False),
        )
        # replace_bn_with_gn(self.fusion_cls)
        #         replace_bn_with_syncbn(self.fusion_cls)

        self.fusion_seg = nn.Sequential(
            # nn.Linear(64 + 768, 256),
            nn.Conv2d(24 + 768, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
        )
        # replace_bn_with_gn(self.fusion_seg)
        #         replace_bn_with_syncbn(self.fusion_seg)

        self.final_interpolate = kw['interpolation']
        self.cls_loss_type = 'ce'
        if self.cls_loss_type == 'ce':
            self.cls_loss = nn.CrossEntropyLoss()
        elif self.cls_loss_type == 'asl':
            self.cls_loss = ASLSingleLabel(gamma_pos=0, gamma_neg=4)

        #         self.vis_file = "./inferencevis_" + str(time.time())
        # os.makedirs(self.vis_file, exist_ok=True)
        self.ac = []
        self.dataframe = []
        self.time = 0

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

    def device(self):
        return 'cuda'

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = batched_inputs['image'].to('cuda')  # [B, 3, H, W]

        images = self.normalize_input(images)

        return images

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):

        # assert 1 == 0

        if not self.training:
            return self.inference(batched_inputs)
        self.time += 1
        assert not torch.jit.is_scripting(), "Scripting for training mode is not supported."
        vis = False

        # texts = [e['text'] for e in batched_inputs]  # [B]
        texts = batched_inputs['text']

        texts_token = clip.tokenize(texts).to('cuda')
        text_features = self.clip_encoder.encode_text(texts_token)  # [B, 768]
        # print(text_features.shape)

        images = self.preprocess_image(batched_inputs).to(torch.float32)  # [B, 3, H, W]
        H, W = images.shape[-2:]
        image_features = self.backbone(images, text_features=text_features)

        # for k in image_features.keys():
        #     print(k, image_features[k].shape)

        '''
        out torch.Size([8, 24, 256, 256])
        0 torch.Size([8, 24, 126, 126])
        1 torch.Size([8, 24, 126, 126])
        2 torch.Size([8, 48, 62, 62])
        3 torch.Size([8, 64, 30, 30])
        4 torch.Size([8, 128, 14, 14])
        5 torch.Size([8, 160, 14, 14])
        6 torch.Size([8, 256, 6, 6])
        7 torch.Size([8, 1280, 6, 6])
        encoder.0 torch.Size([8, 24, 126, 126])
        encoder.1 torch.Size([8, 24, 126, 126])
        encoder.2 torch.Size([8, 48, 62, 62])
        encoder.3 torch.Size([8, 64, 30, 30])
        encoder.4 torch.Size([8, 128, 14, 14])
        encoder.5 torch.Size([8, 160, 14, 14])
        encoder.6 torch.Size([8, 256, 6, 6])
        encoder.7 torch.Size([8, 1280, 6, 6])
        '''
        # assert 1 == 0

        # print(type(image_features)) # dict
        # for k in image_features.keys():
        #     print(k, image_features[k].shape)
        seg_features = image_features[self.feature_seg]  # [B, 64, nh, nw]
        cls_features = image_features[self.feature_cls]  # [B, 2048, nh, nw]
        # print(seg_features.shape, cls_features.shape)

        text_features1 = text_features.unsqueeze(-1).unsqueeze(-1)
        text_features_seg = text_features1.repeat(1, 1, seg_features.shape[-2], seg_features.shape[-1])
        text_features_cls = text_features1.repeat(1, 1, cls_features.shape[-2], cls_features.shape[-1])

        # print(text_features_seg.shape, text_features_cls.shape)
        # print(torch.cat([seg_features, text_features_seg], dim=1).shape, torch.cat([cls_features, text_features_cls], dim=1).shape)
        seg_features = self.fusion_seg(torch.cat([seg_features, text_features_seg], dim=1))
        cls_features = self.fusion_cls(torch.cat([cls_features, text_features_cls], dim=1))

        # cls_logits = self.cls_head(cls_features) # [B, n_class]
        cls_features = self.max_pool(cls_features).squeeze() + self.mean_pool(cls_features).squeeze()
        cls_logits = self.cls_head(cls_features)  # [B, n_class]
        cls_pred = cls_logits.detach().softmax(-1).argmax(-1).squeeze()

        seg_mask_pred = self.seg_head(seg_features)  # [B, 1, nh, nw]
        nh, nw = seg_mask_pred.shape[-2:]
        if nh != H or nw != W:
            seg_mask_pred = F.interpolate(seg_mask_pred, size=(H, W), mode=self.final_interpolate, align_corners=False)

        #         print(cls_logits.shape, seg_mask_pred.shape)

        # seg_mask_gt = torch.stack([e['seg_mask'].unsqueeze(0) for e in batched_inputs]).to('cuda')
        seg_mask_gt = batched_inputs['seg_mask'].to('cuda')
        #         cls_gt = torch.stack([e['labels'] for e in batched_inputs]).to('cuda')
        cls_gt = batched_inputs['labels'].to('cuda')

        ac = cls_pred.cpu().detach().numpy() == cls_gt.cpu().detach().numpy()
        # print(ac, ac.sum(), ac.shape[0])
        ac = ac.sum() / ac.shape[0]
        self.ac.append(ac)
        if self.time>0 and self.time%20==0:
            print("mean accuracy during training: ", np.mean(self.ac))

        if self.time != 0 and self.time % 500 == 0:
            print(cls_gt, cls_logits.detach().softmax(-1).argmax(-1))

        loss_seg = 10 * F.binary_cross_entropy(seg_mask_pred, seg_mask_gt)
        loss_cls = 10 * self.cls_loss(cls_logits, cls_gt)

        losses = {
            'loss_seg': loss_seg,
            'loss_cls': loss_cls,
        }

        # print(losses)

        # assert 1 == 0

        return losses, ac

    @torch.no_grad()
    def inference(
            self,
            batched_inputs: Tuple[Dict[str, torch.Tensor]],
            do_postprocess: bool = True,
    ):

        print("do inference...")
        # assert 1 == 0

        assert not torch.jit.is_scripting(), "Scripting for training mode is not supported."
        # self.model.eval()
        #         texts = [e['text'] for e in batched_inputs]  # [B]
        texts = batched_inputs['text']
        # print(texts)
        texts_token = clip.tokenize(texts).to('cuda')
        text_features = self.clip_encoder.encode_text(texts_token)  # [B, 768]
        # print(text_features.shape)

        images = self.preprocess_image(batched_inputs).to(torch.float32)  # [B, 3, H, W]
        H, W = images.shape[-2:]
        image_features = self.backbone(images, text_features=text_features)

        # print(type(image_features)) # dict
        # for k in image_features.keys():
        #     print(k, image_features[k].shape)
        seg_features = image_features[self.feature_seg]  # [B, 64, nh, nw]
        cls_features = image_features[self.feature_cls]  # [B, 2048, nh, nw]
        # print(seg_features.shape, cls_features.shape)

        text_features1 = text_features.unsqueeze(-1).unsqueeze(-1)
        text_features_seg = text_features1.repeat(1, 1, seg_features.shape[-2], seg_features.shape[-1])
        text_features_cls = text_features1.repeat(1, 1, cls_features.shape[-2], cls_features.shape[-1])

        # print(text_features_seg.shape, text_features_cls.shape)
        # print(torch.cat([seg_features, text_features_seg], dim=1).shape, torch.cat([cls_features, text_features_cls], dim=1).shape)
        seg_features = self.fusion_seg(torch.cat([seg_features, text_features_seg], dim=1))
        cls_features = self.fusion_cls(torch.cat([cls_features, text_features_cls], dim=1))

        # cls_logits = self.cls_head(cls_features) # [B, n_class]
        cls_features = self.max_pool(cls_features).squeeze(-1).squeeze(-1) + self.mean_pool(cls_features).squeeze(
            -1).squeeze(-1)
        # print(cls_features.shape)
        cls_logits = self.cls_head(cls_features)  # [B, n_class]

        seg_mask_pred = self.seg_head(seg_features)  # [B, 1, nh, nw]
        nh, nw = seg_mask_pred.shape[-2:]
        if nh != H or nw != W:
            seg_mask_pred = F.interpolate(seg_mask_pred, size=(H, W), mode=self.final_interpolate, align_corners=False)

        if 'seg_mask' in batched_inputs:
            #             seg_mask_gt = torch.stack([e['seg_mask'].unsqueeze(0) for e in batched_inputs]).to('cuda')
            seg_mask_gt = batched_inputs['seg_mask'].to('cuda')
        else:
            seg_mask_gt = None
        if 'labels' in batched_inputs:
            #             cls_gt = torch.stack([e['labels'] for e in batched_inputs]).to('cuda')
            cls_gt = batched_inputs['labels'].to('cuda')
        else:
            cls_gt = None

        #         self.vis_train(
        #             seg_gts=seg_mask_gt, seg_preds=seg_mask_pred, cls_preds=cls_logits, cls_gts=cls_gt,
        #             ori_images=[e['ori_image'] for e in batched_inputs], texts=[e['dataframe'] for e in batched_inputs],
        #             is_train=False
        #         )

        cls_pred = cls_logits.softmax(-1).argmax(-1).squeeze()
        # dataframe =
        print("#" * 20)
        # print(cls_pred, cls_gt)
        # assert cls_gt is not None
        if cls_gt is not None:
            print(cls_pred, cls_gt)
            ac = cls_pred.cpu().detach().numpy() == cls_gt.cpu().detach().numpy()
            print(ac, ac.sum(), ac.shape[0])
            ac = ac.sum() / ac.shape[0]
            self.ac.append(ac)

            print(np.mean(self.ac))

        #         for idx, e in enumerate(batched_inputs):
        dataframe = batched_inputs['dataframe']
        assert cls_logits.shape[0] == 1
        res = cls_logits.softmax(-1).detach().squeeze().cpu().numpy().tolist()
        dataframe['severity'] = {'normal_mild': res[0], 'moderate': res[1], 'severe': res[2]}
        # print(dataframe)
        dataframe.pop('data_index')
        self.dataframe.append(dataframe)
        # print(self.dataframe)

        with open('./kaggle/res_' + str(self.time) + '.json', 'w') as f:
            json.dump(self.dataframe, f, indent=4)

model = SpineModel(-1)
print(model)

trained_pth = torch.load('/home/kyfq/detectron2/train_spinels_resnet50/model_final.pth')['model']
# model.load_state_dict(trained_pth)
print(trained_pth.keys())
assert 1 == 0
# x = torch.randn((2, 3, 256, 256))
# model(x)
# assert 1 == 0

import os.path as osp
import json
import os
import random
import pandas as pd
import pickle
import numpy as np
import scipy
import pickle

label_map = {'normal_mild': 0, 'moderate': 1, 'severe': 2}
level_map = {
    'l1/l2': 0,
    'l2/l3': 1,
    'l3/l4': 2,
    'l4/l5': 3,
    'l5/s1': 4,
}


def register_cell2(split='train'):
    print("generate dataset for ", split)
    file_path = '/home/kyfq/data/t/rsna-2024-lumbar-spine-degenerative-classification/'

    train = pd.read_csv(file_path + 'train.csv')
    label = pd.read_csv(file_path + 'train_label_coordinates.csv')
    train_desc = pd.read_csv(file_path + 'train_series_descriptions.csv')
    test_desc = pd.read_csv(file_path + 'test_series_descriptions.csv')
    sub = pd.read_csv(file_path + 'sample_submission.csv')

    def generate_image_paths(df, data_dir):
        image_paths = []
        for study_id, series_id in zip(df['study_id'], df['series_id']):
            study_dir = os.path.join(data_dir, str(study_id))
            series_dir = os.path.join(study_dir, str(series_id))
            images = os.listdir(series_dir)
            image_paths.extend([os.path.join(series_dir, img) for img in images])
        return image_paths

    def reshape_row(row):
        data = {'study_id': [], 'condition': [], 'level': [], 'severity': []}

        for column, value in row.items():
            if column not in ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description']:
                parts = column.split('_')
                condition = ' '.join([word.capitalize() for word in parts[:-2]])
                level = parts[-2].capitalize() + '/' + parts[-1].capitalize()
                data['study_id'].append(row['study_id'])
                data['condition'].append(condition)
                data['level'].append(level)
                data['severity'].append(value)

        return pd.DataFrame(data)

    # Define a function to check if a path exists
    def check_exists(path):
        return os.path.exists(path)

    # Define a function to check if a study ID directory exists
    def check_study_id(row):
        study_id = row['study_id']
        path = f'{file_path}/train_images/{study_id}'
        return check_exists(path)

    # Define a function to check if a series ID directory exists
    def check_series_id(row):
        study_id = row['study_id']
        series_id = row['series_id']
        path = f'{file_path}/train_images/{study_id}/{series_id}'
        return check_exists(path)

    # Define a function to check if an image file exists
    def check_image_exists(row):
        image_path = row['image_path']
        return check_exists(image_path)

    # Reshape the DataFrame for all rows
    new_train_df = pd.concat([reshape_row(row) for _, row in train.iterrows()], ignore_index=True)

    # Merge the dataframes on the common columns
    merged_df = pd.merge(new_train_df, label, on=['study_id', 'condition', 'level'], how='inner')
    # Merge the dataframes on the common column 'series_id'
    final_merged_df = pd.merge(merged_df, train_desc, on='series_id', how='inner')

    # Merge the dataframes on the common column 'series_id'
    final_merged_df = pd.merge(merged_df, train_desc, on=['series_id', 'study_id'], how='inner')

    # Create the row_id column
    final_merged_df['row_id'] = (
            final_merged_df['study_id'].astype(str) + '_' +
            final_merged_df['condition'].str.lower().str.replace(' ', '_') + '_' +
            final_merged_df['level'].str.lower().str.replace('/', '_')
    )

    # Create the image_path column
    final_merged_df['image_path'] = (
            f'{file_path}/train_images/' +
            final_merged_df['study_id'].astype(str) + '/' +
            final_merged_df['series_id'].astype(str) + '/' +
            final_merged_df['instance_number'].astype(str) + '.dcm'
    )

    final_merged_df[final_merged_df["severity"] == "Normal/Mild"].value_counts().sum()
    final_merged_df[final_merged_df["severity"] == "Moderate"].value_counts().sum()

    # Define the base path for test images
    base_path = '/home/kyfq/data/t/rsna-2024-lumbar-spine-degenerative-classification/test_images/'

    # Function to get image paths for a series
    def get_image_paths(row):
        series_path = os.path.join(base_path, str(row['study_id']), str(row['series_id']))
        if os.path.exists(series_path):
            return [os.path.join(series_path, f) for f in os.listdir(series_path) if
                    os.path.isfile(os.path.join(series_path, f))]
        return []

    # Mapping of series_description to conditions
    condition_mapping = {
        'Sagittal T1': {'left': 'Left_Neural_Foraminal_Narrowing', 'right': 'Right_Neural_Foraminal_Narrowing'},
        'Axial T2': {'left': 'Left_Subarticular_Stenosis', 'right': 'Right_Subarticular_Stenosis'},
        'Sagittal T2/STIR': 'Spinal_Canal_Stenosis'
    }

    # Create a list to store the expanded rows
    expanded_rows = []
    levels = ['L1_L2', 'L2_L3', 'L3_L4', 'L4_L5', 'L5_S1']

    # Function to update row_id with levels
    def update_row_id(row, levels):
        level = levels[row.name % len(levels)]
        return f"{row['study_id']}_{row['condition']}_{level}"

    # Expand the dataframe by adding new rows for each file path
    for index, row in test_desc.iterrows():
        image_paths = get_image_paths(row)
        conditions = condition_mapping.get(row['series_description'], {})
        if isinstance(conditions, str):  # Single condition
            conditions = {'left': conditions, 'right': conditions}
        for side, condition in conditions.items():
            for image_path in image_paths:
                expanded_rows.append({
                    'study_id': row['study_id'],
                    'series_id': row['series_id'],
                    'series_description': row['series_description'],
                    'image_path': image_path,
                    'condition': condition,
                    'row_id': f"{row['study_id']}_{condition}"
                })

    # Create a new dataframe from the expanded rows
    expanded_test_desc = pd.DataFrame(expanded_rows)
    expanded_test_desc['row_id'] = expanded_test_desc.apply(lambda row: update_row_id(row, levels), axis=1)

    exe = False
    if exe:
        res = json.load(open('./kaggle/res.json', 'r'))
        ans = {}
        for idx in range(len(res)):
            image_path = res[idx]['image_path']
            ans[image_path] = res[idx]['severity']

        sub_data = expanded_test_desc
        sub_data['normal_mild'] = [-1] * len(expanded_test_desc['row_id'])
        sub_data['moderate'] = [-1] * len(expanded_test_desc['row_id'])
        sub_data['severe'] = [-1] * len(expanded_test_desc['row_id'])

        for idx in range(len(expanded_test_desc['row_id'])):
            image_path = expanded_test_desc['image_path'][idx].lower()
            assert image_path in ans, image_path
            severity = ans[image_path]
            sub_data['normal_mild'][idx] = severity['normal_mild']
            sub_data['moderate'][idx] = severity['moderate']
            sub_data['severe'][idx] = severity['severe']

        sub_data.pop('study_id')
        sub_data.pop('series_id')
        sub_data.pop('series_description')
        sub_data.pop('image_path')
        sub_data.pop('condition')

        # print(sub_data)
        for idx in range(len(sub_data['row_id'])):
            for k in sub_data.keys():
                print(k, sub_data[k][idx])
            print("#################################")

        print(sub_data.head(10))
        # Group by 'row_id' and sum the values
        grouped_submission = sub_data.groupby('row_id').sum().reset_index()

        # Normalize the columns
        grouped_submission[['normal_mild', 'moderate', 'severe']] = grouped_submission[
            ['normal_mild', 'moderate', 'severe']].div(
            grouped_submission[['normal_mild', 'moderate', 'severe']].sum(axis=1), axis=0)

        # Check the first 3 rows
        print(grouped_submission.head(3))
        print(len(grouped_submission))
        print(len(sub))
        sub[['normal_mild', 'moderate', 'severe']] = grouped_submission[['normal_mild', 'moderate', 'severe']]
        sub.to_csv("./submission.csv", index=False)
        assert 1 == 0

    # change severity column labels
    # Normal/Mild': 'normal_mild', 'Moderate': 'moderate', 'Severe': 'severe'}
    final_merged_df['severity'] = final_merged_df['severity'].map(
        {'Normal/Mild': 'normal_mild', 'Moderate': 'moderate', 'Severe': 'severe'})

    test_data = expanded_test_desc
    train_data = final_merged_df

    # Apply the functions to the train_data dataframe
    train_data['study_id_exists'] = train_data.apply(check_study_id, axis=1)
    train_data['series_id_exists'] = train_data.apply(check_series_id, axis=1)
    train_data['image_exists'] = train_data.apply(check_image_exists, axis=1)

    # Filter train_data
    train_data = train_data[
        (train_data['study_id_exists']) & (train_data['series_id_exists']) & (train_data['image_exists'])]
    train_data = train_data.dropna()
    train_data = train_data.reset_index(drop=True)

    data_dict = []

    count = {}  # for re-sample

    if split == 'train':
        for idx in range(len(train_data)):
            tmp = {
                'study_id': train_data['study_id'][idx],
                'condition': train_data['condition'][idx],
                'level': train_data['level'][idx].replace("/", " and "),
                'severity': train_data['severity'][idx],
                'series_id': train_data['series_id'][idx],
                'row_id': train_data['row_id'][idx],
                'image_path': train_data['image_path'][idx],
                'x': train_data['x'][idx],
                'y': train_data['y'][idx],

            }
            # tmp['labels'] = np.zeros((3,))
            tmp['labels'] = label_map[tmp['severity']]

            if tmp['condition'] not in count.keys():
                count[tmp['condition']] = {}
                if label_map[tmp['severity']] not in count[tmp['condition']].keys():
                    count[tmp['condition']][label_map[tmp['severity']]] = [idx]
                else:
                    count[tmp['condition']][label_map[tmp['severity']]].append(idx)
            else:
                if label_map[tmp['severity']] not in count[tmp['condition']].keys():
                    count[tmp['condition']][label_map[tmp['severity']]] = [idx]
                else:
                    count[tmp['condition']][label_map[tmp['severity']]].append(idx)

            # if 'spine' in tmp['condition']:
            #
            #     tmp['labels'][0 * 5 + level_map[tmp['level'].lower()]] = label_map[tmp['severity']]
            # elif 'neural' in tmp['condition']:
            #
            #     tmp['labels'][1 * 5 + level_map[tmp['level'].lower()]] = label_map[tmp['severity']]
            # elif 'subarticular' in tmp['condition']:
            #
            #     tmp['labels'][2 * 5 + level_map[tmp['level'].lower()]] = label_map[tmp['severity']]
            tmp['data_index'] = len(data_dict)
            tmp['text'] = tmp['condition'] + ', focus on the area between ' + tmp[
                'level'] + ", and judge the disease severity:  normal_mild (0), 'moderate (1), severe (2)"

            data_dict.append(tmp)

        print("before re-sample...")
        for k in count.keys():
            for k2 in count[k].keys():
                print(k, k2, len(count[k][k2]))
        '''
        Spinal Canal Stenosis 0 8552
        Spinal Canal Stenosis 1 732
        Spinal Canal Stenosis 2 469
        Left Neural Foraminal Narrowing 0 7671
        Left Neural Foraminal Narrowing 1 1792
        Left Neural Foraminal Narrowing 2 397
        Right Neural Foraminal Narrowing 0 7684
        Right Neural Foraminal Narrowing 1 1767
        Right Neural Foraminal Narrowing 2 378
        Left Subarticular Stenosis 0 6857
        Left Subarticular Stenosis 1 1834
        Left Subarticular Stenosis 2 912
        Right Subarticular Stenosis 0 6862
        Right Subarticular Stenosis 1 1825
        Right Subarticular Stenosis 2 925
        '''

        # assert 1 == 0
        # resample
        for k1 in count.keys():  # resample level of each condition

            max_num = int(max([len(v) for k, v in count[k1].items()]) // 2.5)
            for k2 in count[k1].keys():
                now_num = len(count[k1][k2])
                if now_num >= max_num:
                    continue

                cand_idxs = count[k1][k2]
                # print(cand_idxs)
                cand_idxs = np.random.choice(cand_idxs, max_num - now_num + 1)
                for idx in cand_idxs:
                    tmp = {
                        'study_id': train_data['study_id'][idx],
                        'condition': train_data['condition'][idx],
                        'level': train_data['level'][idx].replace("/", " and "),
                        'severity': train_data['severity'][idx],
                        'series_id': train_data['series_id'][idx],
                        'row_id': train_data['row_id'][idx],
                        'image_path': train_data['image_path'][idx],
                        'x': train_data['x'][idx],
                        'y': train_data['y'][idx],

                    }
                    # tmp['labels'] = np.zeros((3,))
                    tmp['labels'] = label_map[tmp['severity']]

                    if tmp['condition'] not in count.keys():
                        count[tmp['condition']] = {}
                        if label_map[tmp['severity']] not in count[tmp['condition']].keys():
                            count[tmp['condition']][label_map[tmp['severity']]] = [idx]
                        else:
                            count[tmp['condition']][label_map[tmp['severity']]].append(idx)
                    else:
                        if label_map[tmp['severity']] not in count[tmp['condition']].keys():
                            count[tmp['condition']][label_map[tmp['severity']]] = [idx]
                        else:
                            count[tmp['condition']][label_map[tmp['severity']]].append(idx)

                    # if 'spine' in tmp['condition']:
                    #
                    #     tmp['labels'][0 * 5 + level_map[tmp['level'].lower()]] = label_map[tmp['severity']]
                    # elif 'neural' in tmp['condition']:
                    #
                    #     tmp['labels'][1 * 5 + level_map[tmp['level'].lower()]] = label_map[tmp['severity']]
                    # elif 'subarticular' in tmp['condition']:
                    #
                    #     tmp['labels'][2 * 5 + level_map[tmp['level'].lower()]] = label_map[tmp['severity']]
                    tmp['data_index'] = len(data_dict)
                    tmp['text'] = tmp['condition'] + ', focus on the area between ' + tmp[
                        'level'] + ", and judge the disease severity:  normal_mild (0), 'moderate (1), severe (2)"

                    data_dict.append(tmp)

        print("after re-sample...")
        '''
        Spinal Canal Stenosis 0 8552
        Spinal Canal Stenosis 1 5702
        Spinal Canal Stenosis 2 5702
        Left Neural Foraminal Narrowing 0 7671
        Left Neural Foraminal Narrowing 1 5115
        Left Neural Foraminal Narrowing 2 5115
        Right Neural Foraminal Narrowing 0 7684
        Right Neural Foraminal Narrowing 1 5123
        Right Neural Foraminal Narrowing 2 5123
        Left Subarticular Stenosis 0 6857
        Left Subarticular Stenosis 1 4572
        Left Subarticular Stenosis 2 4572
        Right Subarticular Stenosis 0 6862
        Right Subarticular Stenosis 1 4575
        Right Subarticular Stenosis 2 4575

        '''
        sums = 0
        for k in count.keys():
            for k2 in count[k].keys():
                print(k, k2, len(count[k][k2]))
                sums += len(count[k][k2])
        print(sums)  # 87800



    elif split == 'test':
        # assert 1 == 0
        for idx in range(len(test_data['study_id'])):
            tmp = {
                'study_id': str(test_data['study_id'][idx]),
                # 'condition': test_data['condition'][idx],
                'series_id': str(test_data['series_id'][idx]),
                'row_id': test_data['row_id'][idx].lower(),
                'image_path': test_data['image_path'][idx],

            }
            condition = ""
            for m, e in enumerate(test_data['condition'][idx].split("_")):
                if m == 0:
                    condition += e
                else:
                    condition += " " + e
            tmp['condition'] = condition
            level = ""
            for m, e in enumerate(test_data['row_id'][idx].split("_")[-2:]):
                if m == 0:
                    level += e
                else:
                    level += " and " + e
            tmp['level'] = level

            tmp['data_index'] = len(data_dict)
            tmp['text'] = tmp['condition'] + ', focus on the area between ' + tmp[
                'level'] + ", and judge the disease severity:  normal_mild (0), 'moderate (1), severe (2)"
            data_dict.append(tmp)

    elif split == 'sub':
        assert 1 == 0

    if split == 'train':
        random.shuffle(data_dict)
    # if len(data_dict) == 0:
    #     data_dict.append({
    #         'image_id': 142,
    #         'ref_id': 0,
    #         'raw_sent': ""
    #     })
    print("data for " + split + ": ", len(data_dict))  # before resample: 48657, after resample: 87796
    # assert 1 == 0
    return data_dict


train_data = register_cell2(split='train')
print(len(train_data))
test_data = register_cell2(split='test')
print(len(test_data))

# datamapper
import this
import scipy.io
import scipy.ndimage
import os
import os.path as osp
import json
import math

from PIL import Image
import cv2
import numpy as np
import random
import pickle
import torch
import matplotlib.pyplot as plt
from os.path import splitext

import seaborn as sns

import matplotlib.pyplot as plt
import os
import time
import glob
import collections
import torch.nn as nn
import torchvision.transforms as transforms
import pydicom as dicom
import matplotlib.patches as patches

from matplotlib import animation, rc
import pandas as pd

import pydicom as dicom  # dicom
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


class SpineDatasetMapper:
    def __init__(self, dataset, is_train=True):
        super().__init__()
        print("for spine")
        # print(cfg.all_iter, cfg.e_sche, cfg.samples, cfg.img_file, cfg.data_file, cfg.transforms, cfg.item)
        # Rebuild augmentations
        # print(cfg)
        # print(cfg.items)
        self.items = len(dataset)
        # print(self.items)
        # assert 1 == 0
        self.split = 'train' if is_train else 'test'
        # self.split = 'train'
        # self.split = 'val'
        # self.split = 'train'
        self.training = is_train

        self.is_cos = False

        print("训练模式") if is_train else print("测试模式")

        # self.items = cfg.item

        # self.file_path = '/home/kyfq/data/rsna-2024-lumbar-spine-degenerative-classification/'

        # self.train = pd.read_csv(self.file_path + 'train.csv')
        # self.label = pd.read_csv(self.file_path + 'train_label_coordinates.csv')
        # self.train_desc = pd.read_csv(self.file_path + 'train_series_descriptions.csv')
        # self.test_desc = pd.read_csv(self.file_path + 'test_series_descriptions.csv')
        # self.sub = pd.read_csv(self.file_path + 'sample_submission.csv')

        self.height = 256
        self.width = 256

        # self.transforms = None if cfg.transforms == 0 else cfg.transforms
        #         color_jitter = transforms.ColorJitter(0.3, 0.3, 0.2)
        self.transform = transforms.Compose([
            # transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),  # Convert back to uint8 for PIL
            transforms.ToPILImage(),
            transforms.Resize((self.height, self.width)),
            transforms.Grayscale(num_output_channels=3),
            # transforms.RandomApply([color_jitter], p=0.3),
            # transforms.RandomGrayscale(p=0.3),
            # GussianBlur(kernel_size=3),
            transforms.ToTensor(),
        ])
        # self.transform = None
        #         self.transform_ori = transforms.Compose([
        #             # transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),  # Convert back to uint8 for PIL
        #             transforms.ToPILImage(),
        #             transforms.Resize((self.height, self.width)),
        #             transforms.Grayscale(num_output_channels=3),

        #         ])

        self.seg_radius = 50
        self.seg_gen_type = 'gaussian'

        self.dataset = dataset

    def __len__(self):
        return self.items

    def generate_mask(self, x, y, radius=20, gen_type='normal'):
        if gen_type == 'normal':
            seg_mask = np.zeros((self.height, self.width), dtype=np.int32)
            # print(x, y)
            cv2.circle(seg_mask, (x, y), radius, 1, -1)
        elif gen_type == 'gaussian':
            # use gaussian map center at (x, y)
            cy, cx = np.ogrid[: self.height, : self.width]
            distance = (cx - x) ** 2 + (cy - y) ** 2
            seg_mask = np.exp(-distance / (2 * radius ** 2))

        return seg_mask.astype(np.float32)

    def generate(self, dataset_dict):
        # print("call cell dataset mapper...")
        dataset_dicts = {}

        image_path = dataset_dict['image_path']
        if self.training:
            labels = dataset_dict['labels']
        else:
            labels = None

        text = dataset_dict['text']
        data_index = dataset_dict['data_index']

        oriimg = load_dicom(image_path)

        img = self.transform(oriimg)

        orih, oriw = oriimg.shape[:2]
        scaleh, scalew = self.height / orih, self.width / oriw

        #         resized_oriimg = self.transform_ori(oriimg)
        if self.training:
            pos_x = dataset_dict['x']
            pos_y = dataset_dict['y']

            pos_x = int(scalew * pos_x)
            pos_y = int(scaleh * pos_y)
            seg_mask = self.generate_mask(pos_x, pos_y, radius=self.seg_radius, gen_type=self.seg_gen_type)

        #         dataset_dicts["width"] = self.width
        #         dataset_dicts["height"] = self.height
        if self.training:
            dataset_dicts['labels'] = torch.tensor(labels, dtype=torch.long)  # [15,]

        dataset_dicts['image'] = img  # [3, H, W]
        #         dataset_dicts['ori_image'] = resized_oriimg # [H, W]
        dataset_dicts['text'] = text
        if self.training:
            dataset_dicts['seg_mask'] = torch.tensor(seg_mask).unsqueeze(0)
        else:
            dataset_dicts['dataframe'] = dataset_dict

        # plt.subplot(1, 2, 1)
        # plt.imshow(img)
        # plt.subplot(1, 2, 2)
        # plt.imshow(seg_mask)
        # plt.show()
        # assert 1 == 0

        # assert 1 == 0
        return dataset_dicts

    def __getitem__(self, index):
        dataset_dict = self.dataset[index]
        res = self.generate(dataset_dict)
        #         print(res.keys())
        return res


def load_dicom(path):  # un-normalized
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


class GussianBlur(object):
    def __init__(self, kernel_size):
        radius = kernel_size // 2
        kernel_size = radius * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1), stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size), stride=1, padding=0, bias=False, groups=3)

        self.k = kernel_size
        self.r = radius
        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radius),
            self.blur_h,
            self.blur_v

        )
        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

dataset_train = SpineDatasetMapper(train_data, is_train=True)
dataset_test = SpineDatasetMapper(test_data, is_train=False)

print(len(dataset_train), len(dataset_test))

from torch.utils.data import DataLoader
train_loader = DataLoader(
    dataset_train,
    num_workers=4,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)
test_loader = DataLoader(
    dataset_test,
    num_workers=1,
    batch_size=1,
    shuffle=False,
    drop_last=False,
)

from typing import Set
def build_optimizer(model, opt_name):
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()

    lr = 0.001
    weight_decay = 1e-4
    momentum = 0.9

    # 使用预训练模型时这个地方需要修改
    for key, value in model.named_parameters(recurse=True):
        if not value.requires_grad:
            #             print("不需要训练部分,", key)
            continue
        # Avoid duplicating parameters
        if value in memo:
            continue
        memo.add(value)

        # if "backbone" in key:
        if "backbone" in key:  # 不微调FPN
            #             print("微调部分", key)
            pass
            # lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER  # fine-tune backbone

        elif "bert" in key:
            if "encoder.layer" not in key:
                continue

        #             print("微调：", key)
        # lr = lr * cfg.SOLVER.TEXTENCODER  # fine-tune bert
        elif "clip" in key:
            if "encoder.layer" not in key:
                continue

        #             print("微调：", key)
        # lr = lr * cfg.SOLVER.TEXTENCODER  # fine-tune clip

        else:
            #             print("不微调,", key)
            pass

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    #         print(key, lr)
    # print(key)
    # 包括：backbone, proposal_generator, roi_heads, textencoder
    # print(params)
    # for p in params:
    #     print(p['lr'])
    # assert 1 == 0

    def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = 1.0
        enable = (
                True
                and 'value' == "full_model"
                and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    optimizer_type = opt_name
    if optimizer_type == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params, lr, momentum=momentum
        )
    elif optimizer_type == "ADAMW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params, lr
        )
    elif optimizer_type == 'ADAM':
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.Adam)(
            params, lr
        )
    elif optimizer_type == 'RADAM':
        # print("use radam")
        # assert 1 == 0
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.RAdam)(
            params, lr
        )
    elif optimizer_type == 'Adaiv2':
        assert 1 == 0
        optimizer = maybe_add_full_model_gradient_clipping(Adai)(
            params, lr
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    #     if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
    #         optimizer = maybe_add_gradient_clipping(cfg, optimizer)

    # print(optimizer)

    return optimizer

optimizer = build_optimizer(model, 'ADAM')
# print(optimizer)

import torch.optim.lr_scheduler as lr_scheduler
# Learning rate scheduler
# scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See :paper:`ImageNet in 1h` for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))

from bisect import bisect_right

from fvcore.common.param_scheduler import (
    CompositeParamScheduler,
    ConstantParamScheduler,
    LinearParamScheduler,
    ParamScheduler,
)

class LRMultiplier(torch.optim.lr_scheduler._LRScheduler):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        multiplier: ParamScheduler,
        max_iter: int,
        last_iter: int = -1,
    ):
        """
        Args:
            optimizer, last_iter: See ``torch.optim.lr_scheduler._LRScheduler``.
                ``last_iter`` is the same as ``last_epoch``.
            multiplier: a fvcore ParamScheduler that defines the multiplier on
                every LR of the optimizer
            max_iter: the total number of training iterations
        """
        if not isinstance(multiplier, ParamScheduler):
            raise ValueError(
                "_LRMultiplier(multiplier=) must be an instance of fvcore "
                f"ParamScheduler. Got {multiplier} instead."
            )
        self._multiplier = multiplier
        self._max_iter = max_iter
        super().__init__(optimizer, last_epoch=last_iter)

    def state_dict(self):
        # fvcore schedulers are stateless. Only keep pytorch scheduler states
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch}

    def get_lr(self) -> List[float]:
        multiplier = self._multiplier(self.last_epoch / self._max_iter)
        return [base_lr * multiplier for base_lr in self.base_lrs]

import bisect
class MultiStepParamScheduler(ParamScheduler):

    def __init__(
        self,
        values: List[float],
        num_updates: Optional[int] = None,
        milestones: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            values: param value in each range
            num_updates: the end of the last range. If None, will use ``milestones[-1]``
            milestones: the boundary of each range. If None, will evenly split ``num_updates``

        For example, all the following combinations define the same scheduler:

        * num_updates=90, milestones=[30, 60], values=[1, 0.1, 0.01]
        * num_updates=90, values=[1, 0.1, 0.01]
        * milestones=[30, 60, 90], values=[1, 0.1, 0.01]
        * milestones=[3, 6, 9], values=[1, 0.1, 0.01]  (ParamScheduler is scale-invariant)
        """
        if num_updates is None and milestones is None:
            raise ValueError("num_updates and milestones cannot both be None")
        if milestones is None:
            # Default equispaced drop_epochs behavior
            milestones = []
            step_width = math.ceil(num_updates / float(len(values)))
            for idx in range(len(values) - 1):
                milestones.append(step_width * (idx + 1))
        else:
            if not (
                isinstance(milestones, Sequence)
                and len(milestones) == len(values) - int(num_updates is not None)
            ):
                raise ValueError(
                    "MultiStep scheduler requires a list of %d miletones"
                    % (len(values) - int(num_updates is not None))
                )

        if num_updates is None:
            num_updates, milestones = milestones[-1], milestones[:-1]
        if num_updates < len(values):
            raise ValueError(
                "Total num_updates must be greater than length of param schedule"
            )

        self._param_schedule = values
        self._num_updates = num_updates
        self._milestones: List[int] = milestones

        start_epoch = 0
        for milestone in self._milestones:
            # Do not exceed the total number of epochs
            if milestone >= self._num_updates:
                raise ValueError(
                    "Milestone must be smaller than total number of updates: "
                    "num_updates=%d, milestone=%d" % (self._num_updates, milestone)
                )
            # Must be in ascending order
            if start_epoch >= milestone:
                raise ValueError(
                    "Milestone must be smaller than start epoch: start_epoch=%d, milestone=%d"
                    % (start_epoch, milestone)
                )
            start_epoch = milestone

    def __call__(self, where: float) -> float:
        if where > 1.0:
            raise RuntimeError(
                f"where in ParamScheduler must be in [0, 1]: got {where}"
            )
        epoch_num = int((where + self.WHERE_EPSILON) * self._num_updates)
        return self._param_schedule[bisect.bisect_right(self._milestones, epoch_num)]

class WarmupParamScheduler(CompositeParamScheduler):
    """
    Add an initial warmup stage to another scheduler.
    """

    def __init__(
        self,
        scheduler: ParamScheduler,
        warmup_factor: float,
        warmup_length: float,
        warmup_method: str = "linear",
    ):
        """
        Args:
            scheduler: warmup will be added at the beginning of this scheduler
            warmup_factor: the factor w.r.t the initial value of ``scheduler``, e.g. 0.001
            warmup_length: the relative length (in [0, 1]) of warmup steps w.r.t the entire
                training, e.g. 0.01
            warmup_method: one of "linear" or "constant"
        """
        end_value = scheduler(warmup_length)  # the value to reach when warmup ends
        start_value = warmup_factor * scheduler(0.0)
        if warmup_method == "constant":
            warmup = ConstantParamScheduler(start_value)
        elif warmup_method == "linear":
            warmup = LinearParamScheduler(start_value, end_value)
        else:
            raise ValueError("Unknown warmup method: {}".format(warmup_method))
        super().__init__(
            [warmup, scheduler],
            interval_scaling=["rescaled", "fixed"],
            lengths=[warmup_length, 1 - warmup_length],
        )

# epoches = 4 * len(train_loader)
epoches = 7
# warm_up_epoches = int(epoches * 0.25)
warm_up_epoches = 1
milestones = [int(epoches*0.5), int(epoches*0.8)]

def build_lr_scheduler(
    optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = "WarmupMultiStepLR"

    if name == "WarmupMultiStepLR":
        steps = [x for x in milestones if x <= epoches]
        if len(steps) != len(milestones):
            print("SOLVER.STEPS contains values larger than SOLVER.MAX_ITER.")

        values = [0.1 ** k for k in range(len(steps) + 1)]

        sched = MultiStepParamScheduler(
            # values=[cfg.SOLVER.GAMMA ** k for k in range(len(steps) + 1)],
            values = values,
            milestones=steps,
            num_updates=epoches,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))

    sched = WarmupParamScheduler(
        sched,
        0.001,
        min(warm_up_epoches / epoches, 1.0),
        'linear',
    )
    return LRMultiplier(optimizer, multiplier=sched, max_iter=epoches)



warm_up_with_multistep_lr = lambda epoch: epoch/warm_up_epoches if epoch <= warm_up_epoches else 0.1**len([m for m in milestones if m <= epoch])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)
# scheduler = build_lr_scheduler(optimizer)

# import tqdm
from tqdm import trange, tqdm
# for i in trange(0, 1000, 4):
#     print(i)
#     pass

print(len(train_loader))
print(torch.initial_seed())
# assert 1 == 0


def save_sub(time_s):
    file_path = '/home/kyfq/data/t/rsna-2024-lumbar-spine-degenerative-classification/'

    train_desc = pd.read_csv(file_path + 'train_series_descriptions.csv')
    test_desc = pd.read_csv(file_path + 'test_series_descriptions.csv')
    sub = pd.read_csv(file_path + 'sample_submission.csv')

    def generate_image_paths(df, data_dir):
        image_paths = []
        for study_id, series_id in zip(df['study_id'], df['series_id']):
            study_dir = os.path.join(data_dir, str(study_id))
            series_dir = os.path.join(study_dir, str(series_id))
            images = os.listdir(series_dir)
            image_paths.extend([os.path.join(series_dir, img) for img in images])
        return image_paths

    def reshape_row(row):
        data = {'study_id': [], 'condition': [], 'level': [], 'severity': []}

        for column, value in row.items():
            if column not in ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description']:
                parts = column.split('_')
                condition = ' '.join([word.capitalize() for word in parts[:-2]])
                level = parts[-2].capitalize() + '/' + parts[-1].capitalize()
                data['study_id'].append(row['study_id'])
                data['condition'].append(condition)
                data['level'].append(level)
                data['severity'].append(value)

        return pd.DataFrame(data)

    # Define a function to check if a path exists
    def check_exists(path):
        return os.path.exists(path)

    # Define a function to check if a study ID directory exists
    def check_study_id(row):
        study_id = row['study_id']
        path = f'{file_path}/train_images/{study_id}'
        return check_exists(path)

    # Define a function to check if a series ID directory exists
    def check_series_id(row):
        study_id = row['study_id']
        series_id = row['series_id']
        path = f'{file_path}/train_images/{study_id}/{series_id}'
        return check_exists(path)

    # Define a function to check if an image file exists
    def check_image_exists(row):
        image_path = row['image_path']
        return check_exists(image_path)

    # Define the base path for test images
    base_path = '/home/kyfq/data/t/rsna-2024-lumbar-spine-degenerative-classification/test_images/'

    # Function to get image paths for a series
    def get_image_paths(row):
        series_path = os.path.join(base_path, str(row['study_id']), str(row['series_id']))
        if os.path.exists(series_path):
            return [os.path.join(series_path, f) for f in os.listdir(series_path) if
                    os.path.isfile(os.path.join(series_path, f))]
        return []

    # Mapping of series_description to conditions
    condition_mapping = {
        'Sagittal T1': {'left': 'Left_Neural_Foraminal_Narrowing', 'right': 'Right_Neural_Foraminal_Narrowing'},
        'Axial T2': {'left': 'Left_Subarticular_Stenosis', 'right': 'Right_Subarticular_Stenosis'},
        'Sagittal T2/STIR': 'Spinal_Canal_Stenosis'
    }

    # Create a list to store the expanded rows
    expanded_rows = []
    levels = ['L1_L2', 'L2_L3', 'L3_L4', 'L4_L5', 'L5_S1']

    # Function to update row_id with levels
    def update_row_id(row, levels):
        level = levels[row.name % len(levels)]
        return f"{row['study_id']}_{row['condition']}_{level}"

    # Expand the dataframe by adding new rows for each file path
    for index, row in test_desc.iterrows():
        image_paths = get_image_paths(row)
        conditions = condition_mapping.get(row['series_description'], {})
        if isinstance(conditions, str):  # Single condition
            conditions = {'left': conditions, 'right': conditions}
        for side, condition in conditions.items():
            for image_path in image_paths:
                expanded_rows.append({
                    'study_id': row['study_id'],
                    'series_id': row['series_id'],
                    'series_description': row['series_description'],
                    'image_path': image_path,
                    'condition': condition,
                    'row_id': f"{row['study_id']}_{condition}"
                })

    # Create a new dataframe from the expanded rows
    expanded_test_desc = pd.DataFrame(expanded_rows)
    expanded_test_desc['row_id'] = expanded_test_desc.apply(lambda row: update_row_id(row, levels), axis=1)

    res = json.load(open('/home/kyfq/detectron2/kaggle/res_'+str(time_s)+'.json', 'r'))
    ans = {}
    for idx in range(len(res)):
        assert len(res[idx]['image_path']) == 1
        image_path = res[idx]['image_path'][0]
        ans[image_path] = res[idx]['severity']

    sub_data = expanded_test_desc
    sub_data['normal_mild'] = [-1.] * len(expanded_test_desc['row_id'])
    sub_data['moderate'] = [-1.] * len(expanded_test_desc['row_id'])
    sub_data['severe'] = [-1.] * len(expanded_test_desc['row_id'])

    for idx in range(len(expanded_test_desc['row_id'])):
        image_path = expanded_test_desc['image_path'][idx].lower()
        assert image_path in ans, image_path
        severity = ans[image_path]
        # sub_data['normal_mild'][idx] = severity['normal_mild']
        sub_data.loc[idx, 'normal_mild'] = severity['normal_mild']
        # sub_data['moderate'][idx] = severity['moderate']
        sub_data.loc[idx, 'moderate'] = severity['moderate']
        # sub_data['severe'][idx] = severity['severe']
        sub_data.loc[idx, 'severe'] = severity['severe']

    sub_data.pop('study_id')
    sub_data.pop('series_id')
    sub_data.pop('series_description')
    sub_data.pop('image_path')
    sub_data.pop('condition')

    # print(sub_data)
    # for idx in range(len(sub_data['row_id'])):
    #     for k in sub_data.keys():
    #         print(k, sub_data[k][idx])
    #     print("#################################")

    # print(sub_data.head(10))
    # Group by 'row_id' and sum the values
    grouped_submission = sub_data.groupby('row_id').sum().reset_index()

    # Normalize the columns
    grouped_submission[['normal_mild', 'moderate', 'severe']] = grouped_submission[
        ['normal_mild', 'moderate', 'severe']].div(
        grouped_submission[['normal_mild', 'moderate', 'severe']].sum(axis=1), axis=0)

    # Check the first 3 rows
    # print(grouped_submission.head(3))
    # print(len(grouped_submission))
    print(len(sub))
    sub[['normal_mild', 'moderate', 'severe']] = grouped_submission[['normal_mild', 'moderate', 'severe']]
    sub.to_csv("./kaggle/submission.csv", index=False)


model.to('cuda')
batch_size = 8
# print(model.device)
over_ac = []
over_loss = []
start = time.time()
for epoch in range(epoches):
    idx = 0

    for batched_inputs in tqdm(train_loader):
        optimizer.zero_grad()
        #         print(len(batched_inputs))
        loss_dict, ac = model(batched_inputs)
        over_ac.append(ac)
        if idx != 0 and idx % 20 == 0:
            print(loss_dict, scheduler.get_last_lr()[0])
            print("accuracy during training: ", np.mean(over_ac))
            print("mean loss during training: ", np.mean(over_loss))
        loss = sum((i for k, i in loss_dict.items() if (i is not None and not k.startswith('_'))))
        over_loss.append(loss.detach().cpu().numpy())

        loss.backward()
        optimizer.step()
        idx += 1

        scheduler.step()

        # assert 1 == 0
        # if idx >= 10:
        #     break

    print(scheduler.get_last_lr()[0])
    torch.save(model.state_dict(), './kaggle/model_' + str(model.time) + '.pt')

    model.eval()
    for batched_inputs in tqdm(test_loader):
        model(batched_inputs)

    save_sub(model.time)
    model.train()

end = time.time()
print("all time: ", (end-start)/3600)
