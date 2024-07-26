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

from model.commons import TwoConvNormRelu, ResBlock, Normalize, _ni_3d
from .resnet import *
from .convnext import *
from util.util import lookup_nn, get_nd_max_pool, get_nd_conv, get_nd_linear, update_dict_, get_nn

__all__ = []


def register(obj):
    __all__.append(obj.__name__)
    return obj



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
        self.fusion_type = kwargs.get("fusion_type", 'fusion')
        if self.use_guild == 'seg' and self.fusion_type!='dattn':
            self.guild_net = GeneralizedUNet(
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

        else:
            print("no guild decoder...")


        self.out_channels = list(self.unet.out_channels_list)  # list(in_channels_list)
        # self.out_strides = kwargs.get('in_stride_list')
        self.nd = nd

    def forward(self, inputs):
        # print("in backbone: unet")
        x = inputs
        # print(x.shape)
        if self.normalize is not None:
            raise
            x = self.normalize(x)

        x = self.body(x)  # encoder results


        # for k, v in x.items(): # 0, 1, 2, 3, 4, 5
        #     print(k, v.shape)
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
        return x_ori, x_guild



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



class U22(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        """U-Net 22.

        U-Net with 22 convolutions on 5 feature resolutions (1, 1/2, 1/4, 1/8, 1/16) and one final output layer.

        References:
            - https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        self.save_hyperparameters()
        super().__init__(
            UNetEncoder(in_channels=in_channels, block_cls=block_cls, nd=nd,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs
        )



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



class SlimU22(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        """Slim U-Net 22.

        U-Net with 22 convolutions on 5 feature resolutions (1, 1/2, 1/4, 1/8, 1/16) and one final output layer.
        Like U22, but number of feature channels reduce by half.

        References:
            - https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function. Only used if ``out_channels > 0``.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        self.save_hyperparameters()
        super().__init__(
            UNetEncoder(in_channels=in_channels, base_channels=32, block_cls=block_cls, nd=nd,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs
        )



class WideU22(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        """Slim U-Net 22.

        U-Net with 22 convolutions on 5 feature resolutions (1, 1/2, 1/4, 1/8, 1/16) and one final output layer.
        Like U22, but number of feature channels doubled.

        References:
            - https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function. Only used if ``out_channels > 0``.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        self.save_hyperparameters()
        super().__init__(
            UNetEncoder(in_channels=in_channels, base_channels=128, block_cls=block_cls, nd=nd,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs
        )



class U17(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        """U-Net 17.

        U-Net with 17 convolutions on 4 feature resolutions (1, 1/2, 1/4, 1/8) and one final output layer.

        References:
            - https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function. Only used if ``out_channels > 0``.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        self.save_hyperparameters()
        super().__init__(
            UNetEncoder(in_channels=in_channels, depth=4, block_cls=block_cls, nd=nd,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs
        )



class U12(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        """U-Net 12.

        U-Net with 12 convolutions on 3 feature resolutions (1, 1/2, 1/4) and one final output layer.

        References:
            - https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function. Only used if ``out_channels > 0``.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        self.save_hyperparameters()
        super().__init__(
            UNetEncoder(in_channels=in_channels, depth=3, block_cls=block_cls, nd=nd,
                        **_default_unet_kwargs(backbone_kwargs, pretrained)),
            out_channels=out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs
        )


def _default_res_kwargs(backbone_kwargs, pretrained=False):
    kw = dict(fused_initial=False, pretrained=pretrained)
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw



class ResNet18UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        """ResNet 18 U-Net.

        A U-Net with ResNet 18 encoder.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels. If set to 0, the output layer is omitted.
            final_activation: Final activation function. Only used if ``out_channels > 0``.
            backbone_kwargs: Keyword arguments for encoder.
            pretrained: Whether to use a pretrained encoder. If True default weights are used.
                Alternatively, ``pretrained`` can be a URL of a ``state_dict`` that is hosted online.
            block_cls: Module class that defines a convolutional block. Default: ``TwoConvNormRelu``.
            **kwargs: Additional keyword arguments for ``cd.models.UNet``.
        """
        self.save_hyperparameters()
        super().__init__(ResNet18(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)



class ResNet34UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ResNet34(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNet 34')



class ResNet50UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ResNet50(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNet 50')



class ResNet101UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ResNet101(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNet 101')



class ResNet152UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ResNet152(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNet 152')



class ResNeXt50UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(
            ResNeXt50_32x4d(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
            out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNeXt 50')



class ResNeXt101UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(
            ResNeXt101_32x8d(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
            out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNeXt 101')



class ResNeXt152UNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=False,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(
            ResNeXt152_32x8d(in_channels, nd=nd, **_default_res_kwargs(backbone_kwargs, pretrained)),
            out_channels, final_activation=final_activation, block=block_cls, nd=nd, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ResNeXt 152')




def _default_convnext_kwargs(backbone_kwargs, pretrained=False):
    kw = dict(pretrained=pretrained)
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw



class ConvNeXtSmallUNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=True,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ConvNeXtSmall(in_channels, nd=nd, **_default_convnext_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ConvNeXt Small')



class ConvNeXtLargeUNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=True,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ConvNeXtLarge(in_channels, nd=nd, **_default_convnext_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ConvNeXt Large')



class ConvNeXtBaseUNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=True,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ConvNeXtBase(in_channels, nd=nd, **_default_convnext_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ConvNeXt Base')



class ConvNeXtTinyUNet(UNet):
    def __init__(self, in_channels, out_channels, final_activation=None, backbone_kwargs=None, pretrained=True,
                 block_cls=None, nd=2, **kwargs):
        self.save_hyperparameters()
        super().__init__(ConvNeXtTiny(in_channels, nd=nd, **_default_convnext_kwargs(backbone_kwargs, pretrained)),
                         out_channels, final_activation=final_activation, block=block_cls, **kwargs)

    __init__.__doc__ = ResNet18UNet.__init__.__doc__.replace('ResNet 18', 'ConvNeXt Tiny')


def _default_timm_kwargs(backbone_kwargs, pretrained=False):
    kw = dict(pretrained=pretrained)
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw



def _default_smp_kwargs(backbone_kwargs, pretrained=False):
    if pretrained is True:
        pretrained = 'imagenet'
    elif pretrained is False:
        pretrained = None
    kw = dict(weights=pretrained)
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw




import copy
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth

from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface

from backbone.unet import UNet

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
            print(x.shape)
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


# @register_model()
# @handle_legacy_interface(weights=("pretrained", EfficientNet_V2_S_Weights.IMAGENET1K_V1))
def efficientnet_v2_s(
        *, weights=None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    """
    Constructs an EfficientNetV2-S architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.efficientnet.EfficientNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.EfficientNet_V2_S_Weights
        :members:
    """
    # weights = EfficientNet_V2_S_Weights.verify(weights)
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