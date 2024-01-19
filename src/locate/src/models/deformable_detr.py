# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from utils import box_ops, segment_utils
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

#from .backbone import build_backbone
from .joiner import build_joiner
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
from .matcher import NMS
import copy
from typing import Dict, List
import numpy as np


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, dim_feedforward, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, position_embedding='learned', variant='poseFormer'):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.variant = variant
        if self.variant == 'PoseFormer':
            from models.model_poseformer import PoseTransformer
            poseFormer_num_heads = 3
            poseFormer_embed_dim_ratio = 3  # 25
            self.temp_encoder = PoseTransformer(num_frame=100, num_joints=200, in_chans=3, embed_dim_ratio=poseFormer_embed_dim_ratio, depth=0,
                                           num_heads=poseFormer_num_heads, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                           drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None)
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = backbone.num_channels[i]
                if position_embedding == 'nerf':
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        # nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(20, hidden_dim),
                    ))
                else:
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        # nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
            for i in range(num_feature_levels - num_backbone_outs):
                if position_embedding == 'nerf':
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        # nn.Conv1d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(20, hidden_dim),
                    ))
                else:
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        # nn.Conv1d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            if position_embedding == 'nerf':
                self.input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        # nn.Conv1d(dim_feedforward, hidden_dim, kernel_size=1),
                        # nn.GroupNorm(32, hidden_dim),
                        nn.GroupNorm(20, hidden_dim),
                    )])
            else:
                self.input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        # nn.Conv1d(dim_feedforward, hidden_dim, kernel_size=1),
                        # nn.GroupNorm(32, hidden_dim),
                        nn.GroupNorm(32, hidden_dim),
                    )])

        self.joiner = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    # def forward(self, samples: NestedTensor):
    def forward(self, samples, mask, fixed_position_emb=None):
        """The forward expects a NestedTensor, which consists of:
               # - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               # - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
               - samples.tensor: batched images, of shape [batch_size x 3 x H]
               - samples.mask: a binary mask of shape [batch_size x H], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_segments": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # print('samples.shape, mask.shape: ', samples.shape, mask.shape)  # samples.shape, mask.shape:  torch.Size([2, 600, 100]) torch.Size([2, 100])
        features, pos = self.joiner(samples, mask)
        # pos = torch.zeros_like(pos).to(pos.device)
        # print('pos.shape: ', pos.shape)  # v0: pos.shape:  torch.Size([2, 256, 100])
        # features = features[:, :, :, None]
        # mask = mask[:, :, None]
        if self.variant in ['PoseFormer']:
            features = self.temp_encoder(features)
        if not isinstance(features, NestedTensor):
            # features_list = [nested_tensor_from_tensor_list(f[None]) for f in features]
            features = nested_tensor_from_tensor_list(features)
            pos = [pos]
            # out: List[NestedTensor] = []
            # out.append(features)
            # features = out
        srcs = []
        masks = []
        for l, feat in enumerate([features]):
            src, mask = feat.decompose()
            mask = mask[:, :, None]
            # srcs.append(self.input_proj[l](src))
            # masks.append(mask)
            srcs.append(self.input_proj[l](src[:, :, :, None]))
            masks.append(mask)
            assert mask is not None
        # for l, (each_src, each_mask) in enumerate(zip(samples, mask)):
        #     # src, mask = feat.decompose()
        #     srcs.append(self.input_proj[l](each_src))
        #     masks.append(each_mask)
        #     assert each_mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            # print('_len_srcs: {}'.format(_len_srcs))  # _len_srcs: 1
            pos_list = [i for i in pos]
            for l in range(_len_srcs, self.num_feature_levels):
                # print('l: {}'.format(l))  # l: 1
                if l == _len_srcs:
                    src = self.input_proj[l]([features][-1].tensors[:, :, :, None])
                else:
                    src = self.input_proj[l](srcs[-1])
                # m = samples.mask
                m = mask
                # print('m.shape: {}, m[None].shape: {}, m[:, :, None]: {}, src.shape: {}'
                #       .format(m.shape, m[None].shape, m[:, :, None].shape, src.shape))
                # m.shape: torch.Size([2, 100]), m[None].shape: torch.Size([1, 2, 100]),
                # m[:, :, None]: torch.Size([2, 100, 1]), src.shape: torch.Size([2, 256, 50, 1])
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                # pos_l = self.joiner[0](NestedTensor(src, mask)).to(src.dtype)
                pos_l = self.joiner[0](src, mask).to(src.dtype)
                # print('pos_l.shape: ', pos_l.shape)
                # pos_l.shape:  torch.Size([2, 256, 1])
                srcs.append(src)
                masks.append(mask)
                pos_list.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_coord = outputs_coord[:, :, :, :2]
        # print('outputs_coord.shape: ', outputs_coord.shape) # outputs_coord.shape:  torch.Size([4, 2, 100, 4])

        out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            # print('enc_outputs_coord.shape: ', enc_outputs_coord.shape) # enc_outputs_coord.shape:  torch.Size([2, 100, 4])
            enc_outputs_coord = enc_outputs_coord[:, :, :2]
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_segments': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_segments': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, use_CB_loss=False, beta=0.9999):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.use_CB_loss = use_CB_loss
        self.beta = beta
        self.samples_per_cls = [6131, 284, 467, 1003, 897, 586, 251, 1440, 249, 268, 813, 2687, 618, 5748, 678, 127, 663, 347, 877, 442]

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, version='v1'):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        if self.use_CB_loss:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * self.num_classes
            # print('target_classes_o.shape: {}'.format(target_classes_o.shape))
            labels_one_hot = F.one_hot(target_classes_o, self.num_classes).float().cuda(src_logits.device)

            weights = torch.tensor(weights).float().cuda(src_logits.device)
            if version == 'v0':
                weights = weights.unsqueeze(0) #.unsqueeze(0)
                # print('target_classes_onehot.shape: {}, weights.shape: {}, labels_one_hot.shape: {}'
                #       .format(target_classes_onehot.shape, weights.shape, labels_one_hot.shape))
                # target_classes_onehot.shape: torch.Size([2, 100, 20]), weights.shape: torch.Size([1, 20]), labels_one_hot.shape: torch.Size([21, 20])
                weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
                weights = weights.sum(1)
                weights = weights.unsqueeze(1)
                weights = weights.repeat(1, self.num_classes)
            elif version == 'v1':
                weights = weights.unsqueeze(0).unsqueeze(0)
                # print('target_classes_onehot.shape: {}, weights.shape: {}, labels_one_hot.shape: {}'
                #       .format(target_classes_onehot.shape, weights.shape, labels_one_hot.shape))
                # target_classes_onehot.shape: torch.Size([2, 100, 20]), weights.shape: torch.Size([1, 1, 20]), labels_one_hot.shape: torch.Size([2, 20])
                weights = weights.repeat(target_classes_onehot.shape[0], target_classes_onehot.shape[1], 1) * target_classes_onehot
                weights = weights.sum(2)
                weights = weights.unsqueeze(2)
                weights = weights.repeat(1, 1, self.num_classes)
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=weights,
                                         gamma=2) * src_logits.shape[1]
        else:
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_segments' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_segments_orig = outputs['pred_segments']
        src_segments = src_segments_orig[idx]
        src_segments_scaled = src_segments_orig * torch.cat([t['length'] for t in targets], dim=0).unsqueeze(1).repeat(
            1, 2)[:, None, :]
        src_segments_scaled = src_segments_scaled[idx]

        target_boxes = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_segments_scaled = torch.cat([t['segments'][i] * t['length'] for t, (_, i) in zip(targets, indices)],
                                           dim=0)

        loss_bbox = F.l1_loss(src_segments, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        #     box_ops.box_cxcywh_to_xyxy(src_boxes),
        #     box_ops.box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = 1 - torch.diag(segment_utils.generalized_segment_iou(target_segments_scaled, src_segments_scaled))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class Deformable_PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_segments']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

class PostProcessMatched(nn.Module):
    def __init__(self, matcher, args):
        super().__init__()
        self.matcher = matcher
        self.nms = NMS()
        self.args = args

    @torch.no_grad()
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs, targets, target_lengths, given_class=None, cur_thresh=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        if self.args.use_nms:  # False and
            indices = self.nms(outputs_without_aux, given_class=given_class, cur_thresh=cur_thresh)
        else:
            indices = self.matcher(outputs_without_aux, targets)
        # print('indices: ', indices)
        if targets[0]['segments'].dim() > 1:
            out_logits = outputs['pred_logits']
            # print('prob1: ', out_logits.shape)  # prob1:  torch.Size([1, 100, 20])
            # prob = out_logits.view(out_logits.shape[0], -1)
            # print('prob2: ', prob.shape)  # prob2:  torch.Size([1, 2000])
            prob = F.softmax(out_logits, -1)
            scores, labels = prob[..., :-1].max(-1)

            out_segments = outputs['pred_segments']
            # print('out_segments: ', out_segments.shape)  # out_segments:  torch.Size([1, 100, 2])
            scale_factor = target_lengths
            segments = out_segments * scale_factor[:, None, :]

            idx = self._get_src_permutation_idx(indices)
            out_scores = []
            out_labels = []
            out_segments = []

            for i in list(set(idx[0].tolist())):
                logit_idx = [logit for bidx, logit in zip(idx[0].tolist(), idx[1].tolist()) if bidx == i]
                segment_idx = [segment for bidx, segment in zip(idx[0].tolist(), idx[1].tolist()) if bidx == i]
                out_scores.append(scores[i][logit_idx])
                out_labels.append(labels[i][logit_idx])
                out_segments.append(segments[i][segment_idx, :])
            # print('outputs.keys(): ', outputs.keys()) # dict_keys(['pred_logits', 'pred_segments', 'enc_outputs'])
            # if 'edges' in outputs.keys():
            #     if type(outputs['edges']) != type(None):
            # edges_orig = outputs['edges']
            # output_edges = []
            # for i in list(set(idx[0].tolist())):
            #     edge_ids = [src_index for bidx, src_index in zip(idx[0].tolist(), idx[1].tolist()) if bidx == i]
                # output_edges.append(edges_orig[i][edge_ids, :][:, edge_ids])

            results = [{'score': torch.tensor(scr), 'label': torch.tensor(lbl), 'segment': torch.tensor(seg),
                        } for scr, lbl, seg in
                       zip(out_scores, out_labels, out_segments)]

            # results = [{'score': torch.tensor(scr), 'label': torch.tensor(lbl), 'segment': torch.tensor(seg),
            #             'edge': torch.tensor(e)} for scr, lbl, seg, e in
            #            zip(out_scores, out_labels, out_segments, output_edges)]

            # else:
            #     results = [{'score': torch.empty(targets[0]['segments'].size()),
            #                 'label': torch.empty(targets[0]['segments'].size()),
            #                 'segment': torch.empty(targets[0]['segments'].size()),
            #                 'edge': torch.empty(targets[0]['segments'].size())}]

        else:
            results = [{'score': torch.empty(targets[0]['segments'].size()),
                        'label': torch.empty(targets[0]['segments'].size()),
                        'segment': torch.empty(targets[0]['segments'].size()),
                        'edge': torch.empty(targets[0]['segments'].size())}]
        # print('results: score: {}, label: {}, segment: {}'.format(results[0]['score'], results[0]['label'], results[0]['segment']))
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     num_classes = 250
    num_classes = args.num_classes
    device = torch.device(args.device)

    # backbone = build_backbone(args)
    backbone = build_joiner(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        dim_feedforward=args.dim_feedforward,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        position_embedding=args.position_embedding,
        variant=args.variant,
    ).to(device)
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,
                             beta=args.beta)
    criterion.to(device)
    # postprocessors = {'bbox': PostProcess()}
    # if args.masks:
    #     postprocessors['segm'] = PostProcessSegm()
    #     if args.dataset_file == "coco_panoptic":
    #         is_thing_map = {i: i <= 90 for i in range(201)}
    #         postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)
    postprocessors = {'segments': PostProcessMatched(matcher, args)}
    return model, criterion, postprocessors