"""
Joiner modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from .position_encoding import build_position_encoding
from utils.misc import NestedTensor, is_main_process
import numpy as np

class Joiner(nn.Sequential):
    def __init__(self, position_embedding, args):
        super().__init__(position_embedding)
        self.num_channels = [600]
        self.strides = [2048]

    def forward(self, x, mask):
        pos = self[0](x, mask)
        # print('x.shape, pos.shape: ', x.shape, pos.shape)  # torch.Size([2, 600, 100]) torch.Size([2, 256, 100])
        return x, pos

class Nerf_Joiner(nn.Sequential):
    def __init__(self, position_embedding, embeddirs_fn, netchunk=1024):
        super().__init__()
        # self.num_channels = [600]
        self.strides = [2048]
        self.num_channels = [100]
        self.netchunk = netchunk
        self.position_embedding = position_embedding
        self.embeddirs_fn = embeddirs_fn
        self.normalize = torch.from_numpy(np.load('data/babel/normalize.npy')+1e-9).cuda().unsqueeze(0).unsqueeze(-1).float()
        print('self.normalize.shape: ', self.normalize.shape) # torch.Size([1, 1, 600])

    def forward(self, inputs, mask, viewdirs=None):
        # pos = self[0](x, mask)
        # print('inputs.shape: ', inputs.shape)  # inputs.shape:  torch.Size([2, 600, 100])
        inputs = (inputs / self.normalize).permute(0, 2, 1)
        print(inputs.max(), inputs.min())
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        # print('inputs_flat.shape: ', inputs_flat.shape)  # inputs_flat.shape:  torch.Size([1200, 100])
        embedded = self.position_embedding(inputs_flat)
        # print('embedded.shape: ', embedded.shape)  # embedded.shape:  torch.Size([1200, 2100])

        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        # outputs_flat = batchify(fn, self.netchunk)(embedded)
        outputs = torch.reshape(embedded, list(inputs.shape[:-1]) + [embedded.shape[-1]])
        # print('inputs.shape: ', inputs.shape)
        # print('outputs.shape: ', outputs.shape)  # outputs.shape:  torch.Size([2, 600, 2100])
        # return inputs, outputs[:, :256, :100]
        return outputs, outputs

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

class Deformable_Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels
        # print('self.strides, self.num_channels: ', self.strides, self.num_channels)  #  [32] [2048]

    def forward(self, tensor_list: NestedTensor, mask):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:  # 1
            embed_fns.append(lambda x: x)
            out_dim += d
            # print('out_dim 1: ', out_dim)

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)
            # tensor([  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.])
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:  # 20
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
                # print('out_dim 2: ', out_dim)

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        pos = torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        # print('pos.shape2: ', pos.shape)  # pos.shape2:  torch.Size([1200, 2100])
        return pos


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 1,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def build_joiner(args):
    if False:  # args.with_box_refine or args.two_stage:
        position_embedding = build_position_encoding(args)
        train_backbone = args.lr_backbone > 0
        return_interm_layers = args.masks or (args.num_feature_levels > 1)
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
        model = Deformable_Joiner(backbone, position_embedding)
    elif args.position_embedding == 'nerf':
        position_embedding, input_ch = get_embedder(args.multires, args.i_embed)
        embeddirs_fn = None
        if args.use_viewdirs:
            embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
        model = Nerf_Joiner(position_embedding, embeddirs_fn)
    else:
        position_embedding = build_position_encoding(args)
        model = Joiner(position_embedding, args)
    return model
