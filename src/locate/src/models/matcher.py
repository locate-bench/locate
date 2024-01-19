import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
import utils.segment_utils as segment_utils
import torch.nn.functional as F

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_action. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-actions).
    """

    def __init__(self, cost_class: float = 1, cost_segment: float = 1, cost_siou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_segment = cost_segment
        self.cost_siou = cost_siou
        assert cost_class != 0 or cost_segment != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           actions in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_segment = outputs["pred_segments"].flatten(0, 1)  # [batch_size * num_queries, 4]

        scale_factor = torch.stack([t["length"] for t in targets], dim=0)
        out_segment_scaled = out_segment * scale_factor.unsqueeze(1).repeat(1,num_queries,1).flatten(0,1).repeat(1,2)           

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_segment = torch.cat([v["segments"] for v in targets])
        tgt_segment_scaled = torch.cat([v["segments"] * v['length'] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        if tgt_segment.dim() > 1:
            # Compute the L1 cost between segments
            cost_segment = torch.cdist(out_segment, tgt_segment, p=1)
            # Compute the siou cost between segments 
            cost_siou = -segment_utils.generalized_segment_iou(tgt_segment_scaled, out_segment_scaled) 

        else:
            cost_segment = torch.zeros(cost_class.size()).to(cost_class.device)
            cost_siou = torch.zeros(cost_class.size()).to(cost_class.device)


        # Final cost matrix
        C = self.cost_segment * cost_segment + self.cost_class * cost_class + self.cost_siou * cost_siou
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["segments"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # print('[torch.as_tensor(i, dtype=torch.int64) for i in indices]: ', [torch.as_tensor(i, dtype=torch.int64) for i in indices])
        # [torch.as_tensor(i, dtype=torch.int64) for i in indices]:  [tensor([[23, 29, 74, 98], [ 0,  1,  2,  3]])]
        return [torch.as_tensor(i, dtype=torch.int64) for i in indices]

# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
class NMS_v0(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_action. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-actions).
    """

    def __init__(self, overlap: float=0., top_k: int=200):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.overlap = overlap
        self.top_k = top_k

    @torch.no_grad()
    def forward(self, outputs):
        """Apply non-maximum suppression at test time to avoid detecting too many
        overlapping bounding boxes for a given object.
        Args:
            boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
            scores: (tensor) The class predscores for the img, Shape:[num_priors], in our case: [num_priors, num_classes].
            overlap: (float) The overlap thresh for suppressing unnecessary boxes.
            top_k: (int) The Maximum number of box preds to consider.
        Return:
            The indices of the kept boxes with respect to num_priors.
        """

        scores, classes = torch.max(outputs["pred_logits"].flatten(0, 1), -1)  # (max, max_indices)  # [batch_size * num_queries, num_classes]
        boxes = outputs["pred_segments"].flatten(0, 1)  # [batch_size * num_queries, 2]
        # print('scores.shape: {}, boxes.shape: {}, scores[0:10]: {}'
        #       .format(scores.shape, boxes.shape, scores[0:10]))
        # print('boxes[0:10]: {}'
        #       .format(boxes[0:10]))
        # scores.shape: torch.Size([200, 20]), boxes.shape: torch.Size([200, 2])

        keep = scores.new(scores.size(0)).zero_().long()
        if boxes.numel() == 0:
            return keep
        x1 = boxes[:, 0]
        x2 = boxes[:, 1]
        # x2 = boxes[:, 2]
        # y2 = boxes[:, 3]
        area = torch.mul(x2 - x1, 1.)
        v, idx = scores.sort(0)  # sort in ascending order
        # print('v[:10]: {}, idx[:10]: {}, idx.shape: {}'.format(v[:10], idx[:10], idx.shape))
        # I = I[v >= 0.01]
        idx = idx[-self.top_k:]  # indices of the top-k largest vals
        xx1 = boxes.new()
        # yy1 = boxes.new()
        xx2 = boxes.new()
        # yy2 = boxes.new()
        w = boxes.new()
        # h = boxes.new()

        # keep = torch.Tensor()
        count = 0
        while idx.numel() > 0:
            i = idx[-1]  # index of current largest val
            # keep.append(i)
            keep[count] = i
            count += 1
            if idx.size(0) == 1:
                break
            idx = idx[:-1]  # remove kept element from view
            # load bboxes of next highest vals
            torch.index_select(x1, 0, idx, out=xx1)
            # torch.index_select(y1, 0, idx, out=yy1)
            torch.index_select(x2, 0, idx, out=xx2)
            # torch.index_select(y2, 0, idx, out=yy2)
            # store element-wise max with next highest score
            # print('x1[i]: ', x1[i])
            # print('x2[i]: ', x2[i])
            xx1 = torch.clamp(xx1, min=x1[i].cpu().data)
            # yy1 = torch.clamp(yy1, min=y1[i])
            xx2 = torch.clamp(xx2, max=x2[i].cpu().data)
            # yy2 = torch.clamp(yy2, max=y2[i])
            w.resize_as_(xx2)
            # h.resize_as_(yy2)
            w = xx2 - xx1
            # h = yy2 - yy1
            # check sizes of xx1 and xx2.. after each iteration
            w = torch.clamp(w, min=0.0)
            # h = torch.clamp(h, min=0.0)
            inter = w
            # IoU = i / (area(a) + area(b) - i)
            rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
            union = (rem_areas - inter) + area[i]
            IoU = inter/union  # store result in iou
            # keep only elements with an IoU <= overlap
            # print('IoU: {}'.format(IoU))
            idx = idx[IoU.le(self.overlap)]
        keep = keep[keep.gt(0.)].unsqueeze(0)
        # print('keep: {}, count: {}'.format(keep, count))
        # return [torch.as_tensor((i, 0), dtype=torch.int64).unsqueeze(-1) for idx, i in enumerate(keep)]  # , count
        return [torch.cat([torch.zeros_like(keep), keep], dim=0).int()]  # , count

class NMS_v1(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_action. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-actions).
    """

    def __init__(self, overlap: float=0., top_k: int=200):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.overlap = overlap
        self.top_k = top_k
        self.keep_k = 200
        self.num_queries = 100

    @torch.no_grad()
    def forward(self, outputs, class_i=None, cur_thresh=None):
        """Apply non-maximum suppression at test time to avoid detecting too many
        overlapping bounding boxes for a given object.
        Args:
            boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
            scores: (tensor) The class predscores for the img, Shape:[num_priors], in our case: [num_priors, num_classes].
            overlap: (float) The overlap thresh for suppressing unnecessary boxes.
            top_k: (int) The Maximum number of box preds to consider.
        Return:
            The indices of the kept boxes with respect to num_priors.
        """
        keep_list = []
        # pred_logits_list = torch.split(outputs["pred_logits"], self.num_queries)
        # print('outputs["pred_logits"].shape: ', outputs["pred_logits"].shape)  # eval: outputs["pred_logits"].shape:  torch.Size([1, 100, 20])
        for batch_i in range(outputs["pred_logits"].shape[0]):
            pred_logits = outputs["pred_logits"][batch_i]
            # print('pred_logits.shape: ', pred_logits.shape)  # torch.Size([100, 20])
            pred_logits = F.softmax(pred_logits, -1)
            pred_segments = outputs["pred_segments"][batch_i]
            scores, classes = torch.max(pred_logits, -1)  # (max, max_indices)  # [batch_size * num_queries, num_classes]
            boxes = pred_segments  # [batch_size * num_queries, 2]
            # print('scores.shape: {}, boxes.shape: {}, torch.mean(scores): {}, torch.max(scores): {}, torch.min(scores): {}'
            #       .format(scores.shape, boxes.shape, torch.mean(scores), torch.max(scores), torch.min(scores)))
            # scores.shape: torch.Size([100]), boxes.shape: torch.Size([100, 2]), torch.mean(scores)
            # thres:  -2.9599205017089845
            # scores.shape: torch.Size([100]), boxes.shape: torch.Size([100, 2]), torch.mean(scores): 0.2218686193227768
            # print('scores[0:10]: {}'.format(scores[0:10], ))
            # print('boxes[0:10]: {}'.format(boxes[0:10]))
            # scores.shape: torch.Size([100]), boxes.shape: torch.Size([100, 2])
            # scores.shape: torch.Size([200, 20]), boxes.shape: torch.Size([200, 2])
            # thres = np.percentile(scores.cpu().numpy(), 70)
            # print('thres: ', thres)
            # scores = scores[scores.ge(thres)]
            keep = scores.new(scores.size(0)).zero_().long()
            if boxes.numel() == 0:
                return keep
            x1 = boxes[:, 0]
            x2 = boxes[:, 1]
            # x2 = boxes[:, 2]
            # y2 = boxes[:, 3]
            area = torch.mul(x2 - x1, 1.)
            v, original_idx = scores.sort(0)  # sort in ascending order
            # print('v[:10]: {}, idx[:10]: {}, idx.shape: {}'.format(v[:10], idx[:10], idx.shape))
            # I = I[v >= 0.01]
            idx = original_idx[-self.top_k:]  # indices of the top-k largest vals
            xx1 = boxes.new()
            # yy1 = boxes.new()
            xx2 = boxes.new()
            # yy2 = boxes.new()
            w = boxes.new()
            # h = boxes.new()
            classes_selected = classes.new()

            # keep = torch.Tensor()
            count = 0
            while idx.numel() > 0:
                i = idx[-1]  # index of current largest val
                # keep.append(i)
                # print('i: {}, x1[i]: {}, x2[i]: {}'.format(i, x1[i], x2[i]))
                #  x1[i]: 0.8117282390594482, x2[i]: 0.9996324777603149
                # if x2[i] - x1[i] > 0.01:
                #     pass
                keep[count] = i
                count += 1
                if idx.size(0) == 1:
                    break
                idx = idx[:-1]  # remove kept element from view
                # load bboxes of next highest vals
                torch.index_select(x1, 0, idx, out=xx1)
                # torch.index_select(y1, 0, idx, out=yy1)
                torch.index_select(x2, 0, idx, out=xx2)
                torch.index_select(classes, 0, idx, out=classes_selected)
                # torch.index_select(y2, 0, idx, out=yy2)
                # store element-wise max with next highest score
                # print('x1[i]: ', x1[i])
                # print('x2[i]: ', x2[i])
                xx1 = torch.clamp(xx1, min=x1[i].cpu().data)
                # yy1 = torch.clamp(yy1, min=y1[i])
                xx2 = torch.clamp(xx2, max=x2[i].cpu().data)
                # yy2 = torch.clamp(yy2, max=y2[i])
                w.resize_as_(xx2)
                # h.resize_as_(yy2)
                w = xx2 - xx1
                # h = yy2 - yy1
                # check sizes of xx1 and xx2.. after each iteration
                w = torch.clamp(w, min=0.0)
                # h = torch.clamp(h, min=0.0)
                inter = w
                # IoU = i / (area(a) + area(b) - i)
                # rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
                # union = (rem_areas - inter) + area[i]
                # IoU = inter.float() / union.float()  # store result in iou
                # keep only elements with an IoU <= overlap
                # print('IoU: {}, inter: {}, union: {}'.format(IoU, inter, union))
                # torch.equal(classes_selected, classes[i]) and inter > self.overlap
                # idx = idx[torch.logical_or(inter.le(self.overlap), torch.ne(classes_selected, classes[i]))]
                idx = idx[inter.le(self.overlap)]
            keep = keep[keep.gt(0.)].unsqueeze(0)
            # keep = keep.unsqueeze(0)
            keep = keep[:self.keep_k]
            # keep = original_idx.unsqueeze(0)
            # print('keep.shape: ', keep)  # keep.shape:  torch.Size([1, 100])
            keep_list.append(keep)
            # print('keep: {}, count: {}'.format(keep, count))
            # return [torch.as_tensor((i, 0), dtype=torch.int64).unsqueeze(-1) for idx, i in enumerate(keep)]  # , count
        keep_tensor = torch.cat(keep_list, dim=0)
        # print('keep_tensor.shape: ', keep_tensor.shape)
        # # keep_tensor.shape:  torch.Size([1, 100])
        print('[torch.cat([torch.zeros_like(keep_tensor), keep_tensor], dim=0).int()]: ',
              [torch.cat([keep_tensor, torch.zeros_like(keep_tensor)], dim=0).int()])
        return [torch.cat([keep_tensor, torch.zeros_like(keep_tensor)], dim=0).int()]  # , count


class NMS(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_action. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-actions).
    per_class NMS
    """

    def __init__(self, overlap: float=0., top_k: int=200):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.overlap = overlap
        self.top_k = top_k
        self.keep_k = 200
        self.num_queries = 100

    @torch.no_grad()
    def forward(self, outputs, given_class=None, cur_thresh=None, class_wise_thres=None):
        """Apply non-maximum suppression at test time to avoid detecting too many
        overlapping bounding boxes for a given object.
        Args:
            boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
            scores: (tensor) The class predscores for the img, Shape:[num_priors], in our case: [num_priors, num_classes].
            overlap: (float) The overlap thresh for suppressing unnecessary boxes.
            top_k: (int) The Maximum number of box preds to consider.
        Return:
            The indices of the kept boxes with respect to num_priors.
        """
        keep_list = []
        for batch_i in range(outputs["pred_logits"].shape[0]):
            pred_logits = outputs["pred_logits"][batch_i]
            # print('pred_logits.shape: ', pred_logits.shape)  # torch.Size([100, 20])
            pred_logits = F.softmax(pred_logits, -1)
            pred_segments = outputs["pred_segments"][batch_i]
            scores, classes = torch.max(pred_logits, -1)  # (max, max_indices)  # [batch_size * num_queries, num_classes]
            boxes = pred_segments  # [batch_size * num_queries, 2]
            # print('scores.shape: {}, boxes.shape: {}, torch.mean(scores): {}, torch.max(scores): {}, torch.min(scores): {}'
            #       .format(scores.shape, boxes.shape, torch.mean(scores), torch.max(scores), torch.min(scores)))
            # scores.shape: torch.Size([100]), boxes.shape: torch.Size([100, 2]), torch.mean(scores)
            # thres:  -2.9599205017089845
            # scores.shape: torch.Size([100]), boxes.shape: torch.Size([100, 2]), torch.mean(scores): 0.2218686193227768
            # print('scores[0:10]: {}'.format(scores[0:10], ))
            # print('boxes[0:10]: {}'.format(boxes[0:10]))
            # scores.shape: torch.Size([100]), boxes.shape: torch.Size([100, 2])
            # scores.shape: torch.Size([200, 20]), boxes.shape: torch.Size([200, 2])
            # thres = np.percentile(scores.cpu().numpy(), 70)
            # print('thres: ', thres)
            # scores = scores[scores.ge(thres)]
            keep = scores.new(scores.size(0)).zero_().long() - 1
            if boxes.numel() == 0:
                return keep
            x1 = boxes[:, 0]
            x2 = boxes[:, 1]
            v, original_idx = scores.sort(0)  # sort in ascending order
            idx = original_idx[-self.top_k:]  # indices of the top-k largest vals
            xx1 = boxes.new()
            xx2 = boxes.new()
            w = boxes.new()
            classes_selected = classes.new()

            count = 0
            while idx.numel() > 0:
                i = idx[-1]  # index of current largest val
                # print('given_class: {}, cur_thresh: {}, scores[i]: {}, classes[i]: {}'
                #       .format(given_class, cur_thresh, scores[i], classes[i]))
                if given_class is not None and cur_thresh is not None:
                    if scores[i] >= cur_thresh and classes[i] == given_class:
                        # print('i: {}, scores[i]: {}, classes[i]: {}'.format(i, scores[i], classes[i]))
                        keep[count] = i
                        count += 1
                elif class_wise_thres is not None:
                    if scores[i] >= class_wise_thres[classes[i]]:
                        keep[count] = i
                        count += 1
                else:
                    keep[count] = i
                    count += 1
                if idx.size(0) == 1:
                    break
                idx = idx[:-1]  # remove kept element from view
                # load bboxes of next highest vals
                torch.index_select(x1, 0, idx, out=xx1)
                torch.index_select(x2, 0, idx, out=xx2)
                torch.index_select(classes, 0, idx, out=classes_selected)
                # store element-wise max with next highest score
                xx1 = torch.clamp(xx1, min=x1[i].cpu().data)
                xx2 = torch.clamp(xx2, max=x2[i].cpu().data)
                w.resize_as_(xx2)
                w = xx2 - xx1
                w = torch.clamp(w, min=0.0)
                inter = w
                idx = idx[torch.logical_or(inter.le(self.overlap), torch.ne(classes_selected, classes[i]))]
                # idx = idx[inter.le(self.overlap)]
            keep = keep[keep.gt(-1)].unsqueeze(0)
            # keep = keep.unsqueeze(0)
            keep = keep[:self.keep_k]
            keep_list.append(keep)
        keep_tensor = torch.cat(keep_list, dim=0)
        # print('[torch.cat([torch.zeros_like(keep_tensor), keep_tensor], dim=0).int()]: ',
        #       [torch.cat([keep_tensor, torch.zeros_like(keep_tensor)], dim=0).int()])
        return [torch.cat([keep_tensor, torch.zeros_like(keep_tensor)], dim=0).int()]  # , count

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_segment=args.set_cost_segment, cost_siou= args.set_cost_siou)



