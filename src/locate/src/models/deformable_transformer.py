# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from functools import partial

from utils.misc import inverse_sigmoid, to_2tuple, trunc_normal_, drop_path
from einops import rearrange, reduce, repeat
from models.ops.modules import MSDeformAttn
from models.transformer_layers import GraphSelfAttention, GraphEncoderDecoderAttention
from models.multi_attention_forward import multi_head_attention_forward


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, attention_type='divided_space_time',
                 norm_layer=nn.LayerNorm, drop_rate=0., attn_drop_rate=0., img_size=1, patch_size=1,
                 in_chans=3, num_classes=0, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, num_frames=8,
                 drop_path_rate=0.1, variant='poseFormer', position_embedding='learned'):
        # temporal_deformable=False, graph_self_attention=False, space_time=True,
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.variant = variant
        if self.variant == 'TimeSformer':
            from models.model_timesformer import VisionTransformer
            self.encoder = VisionTransformer(img_size=img_size, num_classes=num_classes,
                                           patch_size=patch_size, embed_dim=d_model, depth=12, num_heads=8, mlp_ratio=4,
                                           qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0.,
                                           attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames,
                                           attention_type=attention_type)
        else:
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                              dropout, activation,
                                                              num_feature_levels, nhead, enc_n_points, variant=variant,
                                                              img_size=img_size, patch_size=patch_size, in_chans=100,
                                                              num_classes=num_classes, embed_dim=d_model, depth=12,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_frames=num_frames,
                                                              attention_type=attention_type)
            self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)


        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
            # self.pos_trans = nn.Linear(256 * 2, 256 * 2)
            # self.pos_trans_norm = nn.LayerNorm(256 * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = int(self.d_model // 2)  # 300 # 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    # def get_valid_ratio(self, mask):
    #     # _, H, W = mask.shape
    #     _, H = mask.shape
    #     # valid_H = torch.sum(~mask[:, :, 0], 1)
    #     valid_H = torch.sum(~mask[:, :], 1)
    #     # valid_W = torch.sum(~mask[:, 0, :], 1)
    #     valid_ratio_h = valid_H.float() / H
    #     # valid_ratio_w = valid_W.float() / W
    #     # valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    #     valid_ratio = torch.stack([valid_ratio_h], -1)
    #     return valid_ratio

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        # print('pos_embeds.shape: {}'.format(pos_embeds[0].shape))
        # pos_embeds.shape: torch.Size([2, 256, 100])
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            # print('src.shape: ', src.shape) # torch.Size([2, 256, 100])
            # bs, c, h = src.shape
            # spatial_shape = (h, )
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            # print('src.shape: ', src.shape)  # src.shape:  torch.Size([2, 100, 256])
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            # pos_embed = pos_embed.transpose(0, 1)
            # print('pos_embed.shape: ', pos_embed.shape) # torch.Size([256, 100])
            # pos_embed = pos_embed.flatten(2) # .transpose(1, 2)
            # print('self.level_embed[lvl]: ', pos_embed.shape, self.level_embed[lvl].view(1, -1).shape)  # self.level_embed[lvl]:  torch.Size([100, 256]) torch.Size([1, 256])
            # lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # print('pos_embed.shape, self.level_embed[lvl].view(1, -1).shape: ', pos_embed.shape, self.level_embed[lvl].view(1, -1).shape)
            # print('pos_embed.shape: {}, self.level_embed[lvl].shape: {}'.format(pos_embed.shape, self.level_embed[lvl].shape))
            # pos_embed.shape: torch.Size([2, 100, 256]), self.level_embed[lvl].shape: torch.Size([256])
            # pos_embed.shape: torch.Size([100, 256]), self.level_embed[lvl].shape: torch.Size([256])
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        if self.variant in ['TimeSformer']:
            memory = self.encoder(src_flatten)
            # memory.shape:  torch.Size([2, 768])
        else:
            memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
            # memory.shape:  torch.Size([2, 100, 256])

        # print('memory.shape: ', memory.shape)
        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            # output_proposals = output_proposals[:, :, :2]
            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            # print('output_proposals.shape, output_memory.shape: ', output_proposals.shape, output_memory.shape)
            # output_proposals.shape, output_memory.shape:  torch.Size([2, 100, 4]) torch.Size([2, 100, 256])
            # print('output_proposals: ', output_proposals)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            # print('topk_coords_unact.shape: ', topk_coords_unact.shape, self.get_proposal_pos_embed(topk_coords_unact).shape, )
            # torch.Size([2, 100, 4]) torch.Size([2, 100, 512])
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            # print('pos_trans_out.shape, c: ', pos_trans_out.shape, c)   # pos_trans_out.shape, c:  torch.Size([2, 100, 512]) 600
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, variant='poseFormer',
                 img_size=224, patch_size=16, in_chans=100, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8,
                 attention_type='divided_space_time',
                 ):
        super().__init__()
        self.variant = variant # = temporal_deformable
        # self.graph_self_attention = graph_self_attention
        if self.variant == 'temporal_deformable':
            self.attention_type = attention_type
            self.depth = depth
            # self.dropout = nn.Dropout(dropout)
            # self.num_classes = num_classes
            self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            num_patches = self.patch_embed.num_patches

            ## Positional Embeddings
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)
            if self.attention_type != 'space_only':
                self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
                self.time_drop = nn.Dropout(p=drop_rate)

            ## Attention Blocks
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    attention_type=self.attention_type)
                for i in range(self.depth)])
            self.norm = norm_layer(embed_dim)

            # Classifier head
            # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)

            ## initialization of temporal attention weights
            if self.attention_type == 'divided_space_time':
                i = 0
                for m in self.blocks.modules():
                    m_str = str(m)
                    if 'Block' in m_str:
                        if i > 0:
                            nn.init.constant_(m.temporal_fc.weight, 0)
                            nn.init.constant_(m.temporal_fc.bias, 0)
                        i += 1
        elif self.variant == 'graph_self_attention':
            self.graph_self_attn = GraphSelfAttention(d_model, n_heads, dropout=dropout)
        elif self.variant == 'STAR':
            # set parameters for network architecture
            self.embedding_size = [32]
            self.output_size = 2
            self.dropout_prob = dropout
            # self.args = args

            self.temporal_encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=8)
            self.temporal_encoder_layer_2 = TransformerEncoderLayer(d_model=d_model * 2, nhead=8)

            self.emsize = d_model  # 32  # embedding dimension
            nhid = 2048  # the dimension of the feedforward network model in TransformerEncoder
            nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            nhead = 8  # the number of heads in the multihead-attention models
            dropout = 0.1  # the dropout value
            # print('emsize: ', emsize) # 256
            self.spatial_encoder_1 = TransformerModel(self.emsize, nhead, nhid, nlayers, dropout)
            self.spatial_encoder_2 = TransformerModel(self.emsize, nhead, nhid, nlayers, dropout)

            self.temporal_encoder_1 = TransformerEncoder(self.temporal_encoder_layer, 1)
            self.temporal_encoder_2 = TransformerEncoder(self.temporal_encoder_layer_2, 1)

            # Linear layer to map input to embedding
            self.input_embedding_layer_temporal = nn.Linear(d_model, d_model)
            self.input_embedding_layer_spatial = nn.Linear(d_model, d_model)

            # Linear layer to output and fusion
            self.output_layer = nn.Linear(d_model*2 + 16, d_model)
            self.fusion_layer = nn.Linear(d_model*2, d_model)

            # ReLU and dropout init
            self.relu = nn.ReLU()
            self.dropout_in = nn.Dropout(self.dropout_prob)
            self.dropout_in2 = nn.Dropout(self.dropout_prob)

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        # self.self_attn = GraphSelfAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def get_st_ed(self, batch_num):
        """
        :param batch_num: contains number of pedestrians in different scenes for a batch
        :type batch_num: list
        :return: st_ed: list of tuple contains start index and end index of pedestrians in different scenes
        :rtype: list
        """
        cumsum = torch.cumsum(batch_num, dim=0)
        st_ed = []
        for idx in range(1, cumsum.shape[0]):
            st_ed.append((int(cumsum[idx - 1]), int(cumsum[idx])))

        st_ed.insert(0, (0, int(cumsum[0])))

        return st_ed

    def get_node_index(self, seq_list):
        """
        :param seq_list: mask indicates whether pedestrain exists
        :type seq_list: numpy array [F, N], F: number of frames. N: Number of pedestrians (a mask to indicate whether
                                                                                            the pedestrian exists)
        :return: All the pedestrians who exist from the beginning to current frame
        :rtype: numpy array
        """
        for idx, framenum in enumerate(seq_list):
            if idx == 0:
                node_indices = framenum > 0
            else:
                node_indices *= (framenum > 0)

        return node_indices

    def update_batch_pednum(self, batch_pednum, ped_list):
        """
        :param batch_pednum: batch_num: contains number of pedestrians in different scenes for a batch
        :type list
        :param ped_list: mask indicates whether the pedestrian exists through the time window to current frame
        :type tensor
        :return: batch_pednum: contains number of pedestrians in different scenes for a batch after removing pedestrian who disappeared
        :rtype: list
        """
        updated_batch_pednum_ = copy.deepcopy(batch_pednum).cpu().numpy()
        updated_batch_pednum = copy.deepcopy(batch_pednum)

        cumsum = np.cumsum(updated_batch_pednum_)
        new_ped = copy.deepcopy(ped_list).cpu().numpy()

        for idx, num in enumerate(cumsum):
            num = int(num)
            if idx == 0:
                updated_batch_pednum[idx] = len(np.where(new_ped[0:num] == 1)[0])
            else:
                updated_batch_pednum[idx] = len(np.where(new_ped[int(cumsum[idx - 1]):num] == 1)[0])

        return updated_batch_pednum

    def mean_normalize_abs_input(self, node_abs, st_ed):
        """
        :param node_abs: Absolute coordinates of pedestrians
        :type Tensor
        :param st_ed: list of tuple indicates the indices of pedestrians belonging to the same scene
        :type List of tupule
        :return: node_abs: Normalized absolute coordinates of pedestrians
        :rtype: Tensor
        """
        node_abs = node_abs.permute(1, 0, 2)
        for st, ed in st_ed:
            mean_x = torch.mean(node_abs[st:ed, :, 0])
            mean_y = torch.mean(node_abs[st:ed, :, 1])

            node_abs[st:ed, :, 0] = (node_abs[st:ed, :, 0] - mean_x)
            node_abs[st:ed, :, 1] = (node_abs[st:ed, :, 1] - mean_y)

        return node_abs.permute(1, 0, 2)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        if self.variant == 'temporal_deformable':
            # print('src1.shape: ', src.shape)  # src2.shape:  torch.Size([2, 100, 256])
            B = src.shape[0]
            # print('x-2.shape: ', src.shape)  # x1.shape:  torch.Size([2, 100, 256])  # x-2.shape:  torch.Size([2, 100, 256])
            x = src[:, :, :, None, None]
            # x = torch.transpose(x, 1, 2)
            # print('x-1.shape: ', x.shape)  # x2.shape:  torch.Size([2, 100, 256, 1, 1])  # b c t h w  # x-1.shape:  torch.Size([2, 256, 100, 1, 1])
            x, T, W = self.patch_embed(x)  # x1:  torch.Size([512, 1, 256])  # (b * t) c h w
            # print('x1: ', x.shape) # x1:  torch.Size([200, 1, 256])
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # x2:  torch.Size([512, 2, 256]) # x2:  torch.Size([200, 2, 256])
            # print('x2: ', x.shape)

            ## resizing the positional embeddings in case they don't match the input at inference
            if x.size(1) != self.pos_embed.size(1):
                pos_embed = self.pos_embed
                cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
                other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
                P = int(other_pos_embed.size(2) ** 0.5)
                H = x.size(1) // W
                other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
                new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
                new_pos_embed = new_pos_embed.flatten(2)
                new_pos_embed = new_pos_embed.transpose(1, 2)
                new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
                x = x + new_pos_embed
            else:
                x = x + self.pos_embed
            # print('x3: ', x.shape) # x3:  torch.Size([512, 2, 256]) # x3:  torch.Size([200, 2, 256])
            x = self.pos_drop(x)
            # print('x4: ', x.shape) # x4:  torch.Size([512, 2, 256]) # x4:  torch.Size([200, 2, 256])

            ## Time Embeddings
            if self.attention_type != 'space_only':
                cls_tokens = x[:B, 0, :].unsqueeze(1)
                x = x[:, 1:]
                x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
                ## Resizing time embeddings in case they don't match
                if T != self.time_embed.size(1):
                    time_embed = self.time_embed.transpose(1, 2)
                    new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                    new_time_embed = new_time_embed.transpose(1, 2)
                    x = x + new_time_embed
                else:
                    x = x + self.time_embed
                x = self.time_drop(x)
                x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)
                x = torch.cat((cls_tokens, x), dim=1)
            # print('x5: ', x.shape)  # x5:  torch.Size([2, 257, 256])  # x5:  torch.Size([2, 101, 256])
            ## Attention blocks
            for blk in self.blocks:
                x = blk(x, B, T, W)
            # print('x6: ', x.shape) # x6:  torch.Size([2, 257, 256])  # x6:  torch.Size([2, 101, 256])
            ### Predictions for space-only baseline
            if self.attention_type == 'space_only':
                x = rearrange(x, '(b t) n m -> b t n m', b=B, t=T)
                x = torch.mean(x, 1)  # averaging predictions for every frame
            # print('x7: ', x.shape)  # x7:  torch.Size([2, 257, 256])  # x7: torch.Size([2, 101, 256])
            src = self.norm(x)
            src = src[:, :100]
        elif self.variant == 'graph_self_attention':
            # print('src.shape: ', src.shape, pos.shape)  # src.shape:  torch.Size([2, 100, 256]) torch.Size([100, 256])
            q = k = self.with_pos_embed(src, pos)
            # print('q.shape: ', q.shape)  # q.shape:  torch.Size([2, 100, 256])
            if False:  # src_mask:
                graph = q.permute(1, 0, 2) * src_mask[:, :, None]
            else:
                graph = q.permute(1, 0, 2)
            # print('graph.shape: ', graph.shape)  # graph.shape:  torch.Size([100, 2, 256])
            adj = (torch.ones((q.size(1), q.size(0), q.size(0))))
            adj = adj.to(q.device)
            # print('adj.shape: ', adj.shape)  # adj.shape:  torch.Size([100, 2, 2])
            src, _ = self.graph_self_attn(graph, src, adj)
            # print('src.shape: ', src.shape)  # src.shape:  torch.Size([100, 2, 256])
            src = src.transpose(0, 1)
        elif self.variant == 'STAR':
            # nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum = inputs
            nodes_abs = src
            # num_Ped = nodes_norm.shape[1]

            # outputs = torch.zeros(nodes_norm.shape[0], num_Ped, 2).cuda()
            # GM = torch.zeros(nodes_norm.shape[0], num_Ped, 32).cuda()

            noise = get_noise((1, 100, 16), 'gaussian')

            # for framenum in range(self.args.seq_length - 1):

                # if framenum >= self.args.obs_length and iftest:
                #
                #     node_index = self.get_node_index(seq_list[:self.args.obs_length])
                #     updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
                #     st_ed = self.get_st_ed(updated_batch_pednum)
                #
                #     nodes_current = outputs[self.args.obs_length - 1:framenum, node_index]
                #     nodes_current = torch.cat((nodes_norm[:self.args.obs_length, node_index], nodes_current))
                #     node_abs_base = nodes_abs[:self.args.obs_length, node_index]
                #     node_abs_pred = shift_value[self.args.obs_length:framenum + 1, node_index] + outputs[
                #                                                                                  self.args.obs_length - 1:framenum,
                #                                                                                  node_index]
                #     node_abs = torch.cat((node_abs_base, node_abs_pred), dim=0)
                #     # We normalize the absolute coordinates using the mean value in the same scene
                #     node_abs = self.mean_normalize_abs_input(node_abs, st_ed)
                #
                # else:
                # node_index = self.get_node_index(seq_list[:framenum + 1])
                # nei_list = nei_lists[framenum, node_index, :]
                # nei_list = nei_list[:, node_index]
                # updated_batch_pednum = self.update_batch_pednum(batch_pednum, node_index)
                # st_ed = self.get_st_ed(updated_batch_pednum)
                # nodes_current = nodes_norm[:framenum + 1, node_index]
                # We normalize the absolute coordinates using the mean value in the same scene
            node_abs = self.mean_normalize_abs_input(nodes_abs, [(0, 100)])
            # print('1 node_abs.shape: ', node_abs.shape) # 1 node_abs.shape:  torch.Size([2, 100, 256])
            # Input Embedding
            # if framenum == 0:
            temporal_input_embedded = self.dropout_in(
                self.relu(self.input_embedding_layer_temporal(node_abs)))
            # print('2 temporal_input_embedded.shape: ', temporal_input_embedded.shape) # 2 temporal_input_embedded.shape:  torch.Size([2, 100, 256])
            # else:
            #     temporal_input_embedded = self.dropout_in(
            #         self.relu(self.input_embedding_layer_temporal(nodes_current)))
            #     temporal_input_embedded[:framenum] = GM[:framenum, node_index]

            spatial_input_embedded_ = self.dropout_in2(self.relu(self.input_embedding_layer_spatial(node_abs)))
            # print('3 spatial_input_embedded_.shape: ', spatial_input_embedded_.shape) # 3 spatial_input_embedded_.shape:  torch.Size([2, 100, 256])
            # spatial_input_embedded = self.spatial_encoder_1(spatial_input_embedded_[-1].unsqueeze(1), mask=None)
            spatial_input_embedded = self.spatial_encoder_1(spatial_input_embedded_, mask=None)
            # print('4 spatial_input_embedded.shape: ', spatial_input_embedded.shape)  # torch.Size([2, 100, 256])
            # spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)  # [-1]
            # print('4.2 spatial_input_embedded.shape: ', spatial_input_embedded.shape)
            temporal_input_embedded_last = self.temporal_encoder_1(temporal_input_embedded)  # [-1]
            # print('5 temporal_input_embedded_last.shape: ', temporal_input_embedded_last.shape)  # 5 temporal_input_embedded_last.shape:  torch.Size([2, 100, 256])
            temporal_input_embedded = temporal_input_embedded  # [:-1]
            # print('6 temporal_input_embedded.shape: ', temporal_input_embedded.shape)  # 6 temporal_input_embedded.shape:  torch.Size([1, 100, 256])
            # fusion_feat = torch.cat((temporal_input_embedded_last, spatial_input_embedded), dim=1)
            fusion_feat = torch.cat((temporal_input_embedded_last, spatial_input_embedded), dim=2)
            # print('6.1 fusion_feat.shape: ', fusion_feat.shape)  # 6.1 fusion_feat.shape:  torch.Size([2, 200, 256])
            fusion_feat = self.fusion_layer(fusion_feat)
            # print('7 fusion_feat.shape: ', fusion_feat.shape)  # 7 fusion_feat.shape:  torch.Size([2, 100, 256])
            # spatial_input_embedded = self.spatial_encoder_2(fusion_feat.unsqueeze(1), mask=None)
            spatial_input_embedded = self.spatial_encoder_2(fusion_feat, mask=None)
            # spatial_input_embedded = spatial_input_embedded.permute(1, 0, 2)
            # print('8 spatial_input_embedded.shape: ', spatial_input_embedded.shape)  # 8 spatial_input_embedded.shape:  torch.Size([2, 100, 256])
            # temporal_input_embedded = torch.cat((temporal_input_embedded, spatial_input_embedded), dim=0)
            temporal_input_embedded = torch.cat((temporal_input_embedded, spatial_input_embedded), dim=2)
            temporal_input_embedded = self.temporal_encoder_2(temporal_input_embedded)  # [-1]
            # print('9 temporal_input_embedded.shape: ', temporal_input_embedded.shape)  # 9 temporal_input_embedded.shape:  torch.Size([2, 100, 512])
            noise_to_cat = noise.repeat(temporal_input_embedded.shape[0], 1, 1)  # 10 noise_to_cat.shape:  torch.Size([2, 256])
            # print('10 noise_to_cat.shape: ', noise_to_cat.shape)
            temporal_input_embedded_wnoise = torch.cat((temporal_input_embedded, noise_to_cat), dim=2)
            # print('11 temporal_input_embedded_wnoise.shape: ', temporal_input_embedded_wnoise.shape) # 11 temporal_input_embedded_wnoise.shape:  torch.Size([2, 100, 528])
            outputs_current = self.output_layer(temporal_input_embedded_wnoise)
            # print('12 outputs_current.shape: ', outputs_current.shape)
            # outputs[framenum, node_index] = outputs_current
            # GM[framenum, node_index] = temporal_input_embedded
            src = outputs_current

        # print('src 0.2.shape: ', src.shape)  # src2.shape:  torch.Size([2, 257, 256]) # torch.Size([2, 101, 256])
        # self attention
        # print('src.shape, pos.shape, reference_points.shape, spatial_shapes.shape, level_start_index.shape',
        #       src.shape, pos.shape, reference_points.shape, spatial_shapes.shape, level_start_index.shape)
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        # print('src2.shape: ', src2.shape)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src

        atts = []

        for i in range(self.num_layers):
            output, attn = self.layers[i](output, src_mask=mask,
                                          src_key_padding_mask=src_key_padding_mask)
            atts.append(attn)
        if self.norm:
            output = self.norm(output)

        return output


class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def forward(self, src, mask):
        if mask:
            print('mask.shape, torch.eye(mask.shape[0], mask.shape[0]).shape: ', mask.shape, torch.eye(mask.shape[0], mask.shape[0]).shape)
            print('src.shape: ', src.shape)
            n_mask = mask + torch.eye(mask.shape[0], mask.shape[0]).cuda()
            n_mask = n_mask.float().masked_fill(n_mask == 0., float(-1e20)).masked_fill(n_mask == 1., float(0.0))
        else:
            n_mask = None
        output = self.transformer_encoder(src, mask=n_mask)

        return output

class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight']

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    # @staticmethod
    # def get_reference_points(spatial_shapes, valid_ratios, device):
    #     reference_points_list = []
    #     # for lvl, (H_, W_) in enumerate(spatial_shapes):
    #     for lvl, (H_, ) in enumerate(spatial_shapes):
    #
    #         # ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
    #                                       # torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
    #         ref_y = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device))[0]
    #         # print('valid_ratios.shape: ', valid_ratios.shape) # valid_ratios.shape:  torch.Size([2, 1, 1])
    #         ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * H_)
    #         # ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
    #         # ref = torch.stack((ref_x, ref_y), -1)
    #         ref = torch.stack((ref_y,), -1)
    #         reference_points_list.append(ref)
    #     reference_points = torch.cat(reference_points_list, 1)
    #     reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    #     return reference_points
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        # self.cross_attn = GraphSelfAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        attention_type=args.attention_type,
        # space_time=args.space_time,
        variant=args.variant,
        # temporal_deformable=args.temporal_deformable,
        # graph_self_attention=args.graph_self_attention,
        position_embedding=args.position_embedding,
    )


