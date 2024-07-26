
import torch
from backbone.unet import EfficientUNet, efficientnet_v2_s


model = efficientnet_v2_s()
print(model)

x = torch.randn((2, 3, 100, 100))
y = model(x)
# print(y.shape)

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

eunet = EfficientUNet(**kw)
print(eunet)
y = eunet(x)
# print(type(y))

# print(len(y)) # [B]
for e in y:
    print(type(e))
    if e is not None:
        print(e.keys())
        for k in e.keys():
            print(k, e[k].shape)

    # print(e)
from os.path import join, isfile, splitext
from torch.hub import load_state_dict_from_url

# def resolve_pretrained(pretrained, state_dict_mapper=None, **kwargs):
#     if isinstance(pretrained, str):
#         if isfile(pretrained):
#             state_dict = torch.load(pretrained)
#         else:
#             state_dict = load_state_dict_from_url(pretrained)
#         if 'state_dict' in state_dict:
#             state_dict = state_dict['state_dict']
#         if '.pytorch.org' in pretrained:
#             if state_dict_mapper is not None:
#                 state_dict = state_dict_mapper(state_dict=state_dict, **kwargs)
#     else:
#         raise ValueError('There is no default set of weights for this model. '
#                          'Please specify a URL or filename using the `pretrained` argument.')
#     return state_dict
#
# state_dict = resolve_pretrained('https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth', in_channels=3,
#                                             )
# model.load_state_dict(torch.load(state_dict))