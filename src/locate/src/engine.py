import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os, sys, json, pickle
import copy
import numpy as np
import math
from typing import Iterable
import time

import utils.misc as utils
import datasets
from metrics.detection_metrics import ActionDetectionEvaluator

def train_one_epoch(epoch, max_norm, model, criterion, data_loader, optimizer, scheduler, device, position_embedding='learned'):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 5

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if position_embedding in ['fixed']:
            fixed_position_emb = []
            for t in targets:
                for k, v in t.items():
                    if k == 'fixed_position_emb':
                        fixed_position_emb.append(v.unsqueeze(0))
            fixed_position_emb = torch.cat(fixed_position_emb).to(device)
            print('fixed_position_emb.shape: ', fixed_position_emb.shape)
        else:
            fixed_position_emb = None

        outputs = model(samples.tensors, samples.mask, fixed_position_emb)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    print("Train epoch:", epoch, "Averaged stats:", train_stats)
    return train_stats



@torch.no_grad()
def evaluate(epoch, model, criterion, postprocessors, data_loader, output_dir, dataset, device, position_embedding='learned', vis=False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('mAP', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Test: [{}]'.format(epoch)
    print_every = 50

    predictions = {}
    groundtruth = {}

    for samples, targets in metric_logger.log_every(data_loader, print_every, header):
        # print('samples: ', samples)
        print(targets[0]['video_id'])
        if vis:
            i = 0
            while os.path.exists('samples_{:06d}.npy'.format(i)):
                i += 1
            save_name = 'samples_{:06d}.npy'.format(i)
            save_dict = {
                            'samples': samples,
                         }
            with open(save_name, 'wb') as f:
                pickle.dump(save_dict, f)
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if position_embedding in ['fixed']:
            fixed_position_emb = []
            for t in targets:
                for k, v in t.items():
                    if k == 'fixed_position_emb':
                        fixed_position_emb.append(v.unsqueeze(0))
            fixed_position_emb = torch.cat(fixed_position_emb).to(device)
            print('fixed_position_emb.shape: ', fixed_position_emb.shape)
        else:
            fixed_position_emb = None

        outputs = model(samples.tensors, samples.mask, fixed_position_emb)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()), **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(mAP=torch.tensor(-1))
        metric_logger.update(classification_mAP=torch.tensor(-1))

        scale_factor = torch.cat([t["length"].data for t in targets], dim=0)
        target_lengths = scale_factor.unsqueeze(1).repeat(1,2)
        results = postprocessors['segments'](outputs, targets, target_lengths)
        # print('results: ', results)

        data_utils = getattr(datasets,dataset+'_utils')

        res = {data_utils.getVideoName(target['video_id'].tolist()): output for target, output in zip(targets, results)}
        gt = {data_utils.getVideoName(target['video_id'].tolist()): target for target in targets}

        predictions.update(res)
        groundtruth.update(gt)
        # break
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    ######For mAP calculation need to gather all data###########
    all_predictions = utils.all_gather(predictions)
    all_groundtruth = utils.all_gather(groundtruth)
    # print('all_predictions: {}, all_groundtruth: {}'.format(all_predictions, all_groundtruth))

    with open(os.path.join(output_dir, '{:02d}_babel_groundtruth_val_agt.pkl'.format(epoch)), "wb") as out:
        pickle.dump(all_groundtruth, out)

    # print(len(all_predictions))
    all_predictions_array_results = []
    for video_dict in all_predictions:
        new_video_dict = {}
        for k, v in video_dict.items():
            # print(v.keys())  # dict_keys(['score', 'label', 'segment', 'edge'])
            array_item = {}
            for key in v.keys():
                if key == 'edge':
                    continue
                array_item[key] = v[key].cpu().numpy().tolist()
            array_item_list = []
            for score, label, segment in zip(array_item['score'], array_item['label'], array_item['segment']):
                array_item_list.append({'score': score, 'label': label, 'segment': segment})
            new_video_dict[k] = array_item_list
        # all_predictions_array_results.append(new_video_dict)
    output_dict = {"version": "Babel", "results": new_video_dict, "external_data": {}}
    with open(os.path.join(output_dir, '{:02d}_babel_detection_val_agt.json'.format(epoch)), "w") as out:
        json.dump(output_dict, out)
    # print('all_predictions: ', all_predictions)
    evaluator = ActionDetectionEvaluator(dataset, all_groundtruth, all_predictions, output_dir)
    detection_stats = evaluator.evaluate()

    # stats = {'mAP': detection_stats}
    stats = detection_stats

    metric_logger.update(**stats)

    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if 'mAP' not in k}
    test_stats.update({k: meter.value for k, meter in metric_logger.meters.items() if 'mAP' in k})

    print("Test epoch:", epoch, "Averaged test stats:",  test_stats)
    return test_stats


@torch.no_grad()
def evaluate_v1(epoch, model, criterion, postprocessors, data_loader, output_dir, dataset, device, position_embedding='learned'):
    '''
    NMS threshold search
    Args:
        epoch:
        model:
        criterion:
        postprocessors:
        data_loader:
        output_dir:
        dataset:
        device:
        position_embedding:

    Returns:

    '''
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('mAP', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Test: [{}]'.format(epoch)
    print_every = 50

    num_step = 20
    num_class = 20
    all_thresh_mAP = np.zeros((num_class, num_step))
    thres_list = np.linspace(0, 1 - 1 / num_step, num_step)
    print(thres_list)
    for class_i in range(0, num_class):
        for thres_i, (cur_thresh) in enumerate(thres_list):
            predictions = {}
            groundtruth = {}

            for samples, targets in metric_logger.log_every(data_loader, print_every, header):
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                if position_embedding in ['fixed']:
                    fixed_position_emb = []
                    for t in targets:
                        for k, v in t.items():
                            if k == 'fixed_position_emb':
                                fixed_position_emb.append(v.unsqueeze(0))
                    fixed_position_emb = torch.cat(fixed_position_emb).to(device)
                    print('fixed_position_emb.shape: ', fixed_position_emb.shape)
                else:
                    fixed_position_emb = None

                outputs = model(samples.tensors, samples.mask, fixed_position_emb)

                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if
                                            k in weight_dict}
                loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
                metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()), **loss_dict_reduced_scaled,
                                     **loss_dict_reduced_unscaled)
                metric_logger.update(class_error=loss_dict_reduced['class_error'])
                metric_logger.update(mAP=torch.tensor(-1))
                metric_logger.update(classification_mAP=torch.tensor(-1))

                scale_factor = torch.cat([t["length"].data for t in targets], dim=0)
                target_lengths = scale_factor.unsqueeze(1).repeat(1, 2)
                results = postprocessors['segments'](outputs, targets, target_lengths, given_class=class_i, cur_thresh=cur_thresh)
                # print('results: ', results)

                data_utils = getattr(datasets, dataset + '_utils')

                res = {data_utils.getVideoName(target['video_id'].tolist()): output for target, output in
                       zip(targets, results)}
                gt = {data_utils.getVideoName(target['video_id'].tolist()): target for target in targets}

                predictions.update(res)
                groundtruth.update(gt)

            # gather the stats from all processes
            metric_logger.synchronize_between_processes()

            ######For mAP calculation need to gather all data###########
            all_predictions = utils.all_gather(predictions)
            all_groundtruth = utils.all_gather(groundtruth)
            # print('all_predictions: {}, all_groundtruth: {}'.format(all_predictions, all_groundtruth))

            with open(os.path.join(output_dir, '{:02d}_babel_groundtruth_val_agt.pkl'.format(epoch)), "wb") as out:
                pickle.dump(all_groundtruth, out)

            # print(len(all_predictions))
            all_predictions_array_results = []
            for video_dict in all_predictions:
                new_video_dict = {}
                for k, v in video_dict.items():
                    # print(v.keys())  # dict_keys(['score', 'label', 'segment', 'edge'])
                    array_item = {}
                    for key in v.keys():
                        if key == 'edge':
                            continue
                        array_item[key] = v[key].cpu().numpy().tolist()
                    array_item_list = []
                    for score, label, segment in zip(array_item['score'], array_item['label'], array_item['segment']):
                        array_item_list.append({'score': score, 'label': label, 'segment': segment})
                    new_video_dict[k] = array_item_list
                # all_predictions_array_results.append(new_video_dict)
            output_dict = {"version": "Babel", "results": new_video_dict, "external_data": {}}
            with open(os.path.join(output_dir, '{:02d}_babel_detection_val_agt.json'.format(epoch)), "w") as out:
                json.dump(output_dict, out)
            # print('all_predictions: ', all_predictions)

            evaluator = ActionDetectionEvaluator(dataset, all_groundtruth, all_predictions, output_dir)
            detection_stats = evaluator.evaluate()

            # stats = {'mAP': detection_stats}
            stats = detection_stats

            metric_logger.update(**stats)

            test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if 'mAP' not in k}
            test_stats.update({k: meter.value for k, meter in metric_logger.meters.items() if 'mAP' in k})

            print("Test epoch:", epoch, "Averaged test stats:",  test_stats)
            print('class_i: {}, thres_i: {}, test_stats[0.50_mAP]: {}'.format(class_i, thres_i, test_stats['0.50_mAP'] * num_class))
            all_thresh_mAP[class_i][thres_i] = test_stats['0.50_mAP'] * num_class
    np.save(os.path.join(output_dir, '{:02d}_all_thresh_mAP.npy'.format(epoch)), all_thresh_mAP)
    print('Class_wise thres: \n{}'.format(np.argmax(all_thresh_mAP, -1)))  # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5]
    return test_stats


@torch.no_grad()
def evaluate_mAP(epoch=9,
                 output_dir='/home/jksun/Program/gtad/packages/activitygraph_transformer/output/'+ \
                            'checkpoints_/checkpoints_numqueries10_lr1e-5_lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_'+ \
                            'lrjoiner1e-5_nheads4_nenclayers4_ndeclayers4_hdim256_sr1_batchsize30_nposembdict512_numinputs100',
                 dataset='babel',
                 save_gtad_prediction_path='/home/jksun/Program/gtad/packages/activitygraph_transformer/output/checkpoints_' + \
                          '/checkpoints_numqueries10_lr1e-5_lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e' + \
                          '-5_nheads4_nenclayers4_ndeclayers4_hdim256_sr1_batchsize30_nposembdict512_numinputs100/09_babel_detection_result_agt_ARPC.pkl'):
    '''
    eval offline:
    python src/engine.py
    Parameters
    ----------
    epoch
    output_dir
    dataset
    save_gtad_prediction_path

    Returns
    -------

    '''
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('mAP', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))

    with open(os.path.join(output_dir, '{:02d}_babel_groundtruth_result_agt.pkl'.format(epoch)), "rb") as out:
        all_groundtruth = pickle.load(out)

    with open(save_gtad_prediction_path, "rb") as out:
        all_predictions = pickle.load(out)  # ['results']

    # print('all_predictions: {}, all_groundtruth: {}'.format(all_predictions, all_groundtruth))

    evaluator = ActionDetectionEvaluator(dataset, all_groundtruth, all_predictions, output_dir)
    detection_stats = evaluator.evaluate()

    # stats = {'mAP': detection_stats}
    stats = detection_stats

    metric_logger.update(**stats)

    # test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if 'mAP' not in k}
    # test_stats.update({k: meter.value for k, meter in metric_logger.meters.items() if 'mAP' in k})
    #
    # print("Test epoch:", epoch, "Averaged test stats:", test_stats)
    return
    

if __name__ == "__main__":
    evaluate_mAP()

