# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import os
import pickle, json
import numpy as np
from evaluate import distance
from argument import get_args
from collections import defaultdict, Counter, OrderedDict
from copy import deepcopy

'''
  downloading problem in mac OSX should refer to this answer:
    https://stackoverflow.com/a/42334357
'''

# configs for histogram
pick_layer = 'avg'  # extract feature of this layer
d_type = 'd1'  # distance type

depth = 1  # retrieved depth, set to None will count the ap for whole database

use_gpu = torch.cuda.is_available()

def get_split_idxs(data):
    '''Return indices of samples for each split (train, val, test).
    Args:
        data: {'X': d.X, 'Psi': d.Psi, 'vids': d.vids, 'fps': d.fps,
                 'skip_cats': d.skip_cats, 'ft_type': d.ft_type}
    Returns:
        split_idxs: {'train': [idx1, idx2, ...], 'val': [...], 'test': [...]}
    '''
    spl_lidxs, splits = defaultdict(list), {}

    # Get video URLs for each split
    # split_path = 'conditional_action_gen/motion-annotation/data'
    split_path = '../../../gtad/gtad_lib/babel'
    with open(os.path.join(split_path, 'babel_splits.json')) as infile:
        splits = json.load(infile)

    # Get indices for split samples
    for idx, vid in enumerate(data['vids']):
        vid_spl = [spl for spl in splits if vid in splits[spl]][0]
        spl_lidxs[vid_spl].append(idx)
    spl_aidxs = {spl: np.array(spl_lidxs[spl]) for spl in spl_lidxs}

    for spl in spl_aidxs:
        print('BABEL dataset # {0} = {1}'.format(spl, len(spl_aidxs[spl])))
    return spl_aidxs


class Database(object):

    def __init__(self, db_dir='../../../AR-Shift-GCN/data/babel/'):
        self.AR_feature_root = os.path.join(db_dir, 'babel_ntu_sk_Nv_100_csz_8_ip_to_AR_feat_ext.npy')
        self.AR_label_root = os.path.join(db_dir, 'babel_seqs_labels.pkl')
        self.num_chunks_per_sequence = 100

        idx2als = {0: ['walk'],
                   1: ['stand'],
                   2: ['turn'],  # Really short action segments?
                   3: ['run', 'jog'],
                   4: ['throw'],
                   5: ['catch'],
                   6: ['jump'],
                   7: ['sit'],
                   8: ['dance'],  # Large diversity expected
                   9: ['grasp object', 'place something', 'take/pick something up', 'move something'],
                   10: ['interact with / use object'],  # Large diversity expected
                   11: ['lift something'],  # Is this distinct enough from take / pick something up? Yeah, I think so.
                   12: ['step'],
                   13: ['stretch', 'yoga'],  # Large diversity expected
                   14: ['squat'],
                   15: ['circular movement'],
                   16: ['bend'],
                   17: ['kick'],
                   18: ['hit', 'punch'],
                   19: ['greet'],
                   }

        # idx2al = {0: 'walk', 1: 'stand', 2: 'turn', 3: 'run', 4: 'step', 5: 'jog', 6: 'throw', 7: 'grasp object',
        #           8: 'grab person / body part', 9: 'jump', 10: 'place something', 11: 'stretch', 12: 'kick',
        #           13: 'circular movement', 14: 'take/pick something up', 15: 'bend', 16: 'sit', 17: 'leave',
        #           18: 'dance', 19: 'interact with / use object'}
        idx2al = {}
        for k, v in idx2als.items():
            idx2al[k] = ' or '.join(v)
        print('idx2al: ', idx2al)
        # raise NotImplementedError
        al2idx = {idx2al[k]: k for k in idx2al}

        spl_X = np.load(self.AR_feature_root)  # spl_X.shape:  (808900, 3, 8, 25, 1)
        spl_X = spl_X.transpose(0, 2, 1, 3, 4)
        spl_X = spl_X.reshape(spl_X.shape[0], -1)
        spl_X = spl_X.reshape(int(spl_X.shape[0] // self.num_chunks_per_sequence), self.num_chunks_per_sequence,
                              spl_X.shape[1])  # spl_X.shape:  (8089, 100, 600)
        print('spl_X.shape: ', spl_X.shape)
        with open('{0}'.format(self.AR_label_root), 'rb') as f:
            data = pickle.load(f)

        Psi = data['Psi']
        spl_Psi_c = []
        spl_Psi_ts = []
        spl_Psi_te = []
        # tmp_ts = {'interact with / use object':[], 'lift something':[]}
        # tmp_te = {'interact with / use object':[], 'lift something':[]}
        for each_psi in Psi:
            each_psi_c = []
            each_psi_ts = []
            each_psi_te = []
            for item in each_psi:
                # print('item: ', item)
                if item['c'] in ['run', 'jog']:
                    action_class = 'run or jog'
                elif item['c'] in ['grasp object', 'place something', 'take/pick something up', 'move something']:
                    action_class = 'grasp object or place something or take/pick something up or move something'
                elif item['c'] in ['stretch', 'yoga']:
                    action_class = 'stretch or yoga'
                elif item['c'] in ['hit', 'punch']:
                    action_class = 'hit or punch'
                else:
                    action_class = item['c']
                if action_class not in al2idx:
                    # print('item[c] {} not in al2idx'.format(item['c']))
                    continue
                each_psi_c.append(al2idx[action_class])
                each_psi_ts.append(item['ts'] * self.num_chunks_per_sequence)
                each_psi_te.append(item['te'] * self.num_chunks_per_sequence)
                # if action_class in ['interact with / use object', 'lift something']:
                # tmp_ts[action_class].append([item['ts'] * num_chunks_per_sequence, item['te'] * num_chunks_per_sequence])
                # tmp_te[action_class].append([item['ts'] * num_chunks_per_sequence, item['te'] * num_chunks_per_sequence])
            spl_Psi_c.append(each_psi_c)
            spl_Psi_ts.append(each_psi_ts)
            spl_Psi_te.append(each_psi_te)
        spl_vids = data['vids']
        ann_idx = data['ann_idx']
        # print('tmp_ts: {}, tmp_te: {}'.format(tmp_ts, tmp_te))
        # raise NotImplementedError

        print('spl_vids.shape: ', len(spl_vids))  # spl_vids.shape:  (27838,)
        print('ann_idx.shape: ', len(ann_idx))
        spl_idxs = get_split_idxs(data)

        subsets = ['train', 'val']
        dataset = OrderedDict()
        idx_video = 0
        num_queries_list = []  # number of action queries, ie detection slot. This is the maximal number of actions that can be detected in a video.
        class_seg_num = {'train': {}, 'val': {}, 'test': {}}
        for subset in subsets:
            action_json = []
            idxs = np.array(spl_idxs[subset])
            for i, (video_name, feat_array, start_array, end_array, action_array, ann_idx_array) \
                    in enumerate(zip(spl_vids, spl_X, spl_Psi_ts, spl_Psi_te, spl_Psi_c, ann_idx)):
                if ann_idx_array not in idxs:
                    continue
                if idx_video % 1000 == 0:
                    print('Getting video %d / %d' % (idx_video, len(spl_vids)), flush=True)
                video_name = video_name.replace('https://crichton.is.tue.mpg.de/', '').replace('/', '-')
                video_name = video_name + '@' + '{0:07d}'.format(ann_idx_array)
                # print('video_name: ', video_name)  # video_name:  hmotionlabeling-012840.mp4@0001844
                if subset in ['train']:
                    subset_name = 'validation'
                else:
                    subset_name = 'test'
                video_name = 'video_{}_{:07d}_{}'.format(subset_name, idx_video, video_name)
                if subset in ['test', 'val']:
                    label_subset = 'testing'
                elif subset in ['train']:
                    label_subset = 'training'
                else:
                    raise NotImplementedError('subset {} is not implemented.'.format(subset))

                action_list = []
                # feat_array = feat_array[None, :].repeat(num_chunks_per_sequence, axis=0)
                for (start_frame, end_frame, act_cat) in zip(start_array, end_array, action_array):
                    # act_cat = 0
                    action_list.append(
                        [int(act_cat), round(start_frame * 1., 1),
                         min(round(feat_array.shape[0] - 1.0, 1), round(end_frame * 1., 1))]
                    )
                    action_list = action_list[:20]
                    if idx2al[act_cat] not in class_seg_num[subset]:
                        class_seg_num[subset][idx2al[act_cat]] = 1
                    else:
                        class_seg_num[subset][idx2al[act_cat]] += 1

                if len(action_list) > 0:
                    # action_list = action_list[:10]
                    num_queries_list.append(len(action_list))
                    # label = {"duration": round(self.num_chunks_per_sequence - 1.0, 2), "subset": label_subset,
                    #          "actions": action_list}
                    # print(label)
                    # action_json.append(deepcopy({'vid': '{}'.format(video_name), 'label': label, 'feat': feat_array}))
                    # print('feat_array.shape: ', feat_array.shape) # (100, 69)
                    # padding_feat_array = np.concatenate((feat_array, np.zeros((feat_array.shape[0], 69 - feat_array.shape[1]))), axis=1)
                    # padding_feat_array = feat_array
                    for chunk_idx, each_chunk in enumerate(feat_array):
                        # print('each_chunk.shape: ', each_chunk.shape, 'video_name: ', video_name, 'each_action_list: ', action_list[0])
                        # each_chunk.shape:  (600,) video_name:  video_test_0006488_hmotionlabeling-009330.mp4@0008072 each_action_list:  [0, 0.0, 99.0]
                        each_chunk_action_json = None
                        for each_action_list in action_list:
                            # print('each_action_list: ', each_action_list)
                            if each_action_list[1] <= chunk_idx <= each_action_list[2]:
                                assert each_action_list[0] < 20, 'each_action_list[0]: {}'.format(each_action_list[0])
                                each_chunk_action_json = {'vid': '{}_{:07d}'.format(video_name, chunk_idx), 'actions': each_action_list[0], 'feat': each_chunk}
                                action_json.append(each_chunk_action_json)
                                # print({'vid': '{}_{:07d}'.format(video_name, chunk_idx), 'actions': each_action_list[0], 'feat': each_chunk})
                                break

                        if each_chunk_action_json is None:
                            each_chunk_action_json = {'vid': '{}_{:07d}'.format(video_name, chunk_idx),
                                                      'actions': 20, 'feat': each_chunk}
                            action_json.append(each_chunk_action_json)


                    # if len(action_json) == 0:
                    #     print('each_action_list: ', each_action_list, chunk_idx)
                    # print('padding_feat_array.shape: ', padding_feat_array.shape)  # (100, 256)
                    # if save_json_and_data:
                    #     np.save(os.path.join(features_root, video_name + '.npy'), padding_feat_array[:, None, None, :])  # (time_step, H, W, feat_dim)
                idx_video = idx_video + 1
            dataset[subset] = action_json
        print('class_seg_num: ', class_seg_num)

        # self.train_dir = os.path.join(db_dir, 'train', '0.pkl')
        # self.val_dir = os.path.join(db_dir, 'val', '0.pkl')
        # self._gen_csv()
        # with open(self.train_dir, 'rb') as f:
        self.train_data = dataset['train']
        # with open(self.val_dir, 'rb') as f:
        self.val_data = dataset['val']

    def __len__(self):
        return len(self.train_data)

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data


if __name__ == "__main__":
    args = get_args()
    # evaluate database
    db = Database()
    # APs = evaluate_class(db, f_class=ResNetFeat, d_type=d_type, depth=depth)
    # cls_MAPs = []
    train_data = db.get_train_data()
    val_data = db.get_val_data()

    ret = {}

    for val_idx, query in enumerate(val_data):
        print('val_idx: {} / {}'.format(val_idx, len(val_data)))
        # print(np.array(query['frames']).shape)
        q_id, q_label, q_feat = query['vid'], query['actions'], query['feat']
        results = []
        for idx, sample in enumerate(train_data):
            s_id, s_label, s_feat = sample['vid'], sample['actions'], sample['feat']
            results.append({
                'vid': q_id,
                'dis': distance(q_feat, s_feat, d_type=d_type),
                'pred_label': s_label,
                'gt_label': q_label
            })
        results = sorted(results, key=lambda x: x['dis'])
        if depth and depth <= len(results):
            results = results[:depth]
        print('results: ', results)

        ret[val_idx] = results
        # if val_idx == 400:
        #   break

    with open('ret.pkl', 'wb') as handle:
        pickle.dump(ret, handle)
        print('Save to {}'.format('ret.pkl'))
    # with open('ret.pkl', 'rb') as handle:
    #     b = pickle.load(handle)

    acc_correct_num = 0
    acc_all_num = 0
    traj_all = 0
    traj_success = 0
    union_len = 0
    intersect_len = 0

    vid_flag = None
    label_start = []
    label_end = []
    # pred_action_list = []
    # gt_action_list = []
    # video_frame_idx = []
    vid_pred_action_list = {}
    vid_gt_action_list = {}
    prev_idx = 0
    detection_dict_gt = {}
    detection_dict_pred = {}
    ret_idx = 0
    for key, dict_value in ret.items():
        ret_idx = ret_idx + 1
        vid = dict_value[0]['vid']
        vid_prefix = '_'.join(vid.split('_')[:-1])
        step_idx = int(vid.split('_')[-1])
        # if vid_prefix != vid_flag:
        if step_idx == 0:
            prev_pred_idx = step_idx
            prev_gt_idx = step_idx
            prev_pred_label = dict_value[0]['pred_label']
            prev_gt_label = dict_value[0]['gt_label']
            if vid_flag is not None:
                detection_dict_gt[vid_flag] = {"subset": "test", "annotations": gt_action_list}
                detection_dict_pred[vid_flag] = pred_action_list
            pred_action_list = []
            gt_action_list = []
            prev_idx = step_idx
            vid_flag = vid_prefix
        else:
            print('prev_pred_label: {}, dict_value[0][pred_label]: {}, prev_pred_label != dict_value[0][pred_label]: {}'
                  .format(prev_pred_label, dict_value[0]['pred_label'], prev_pred_label != dict_value[0]['pred_label']))
            if prev_pred_label != dict_value[0]['pred_label'] or step_idx == 99:  # or ret_idx == len(ret):
                pred_action_list.append({"label": prev_pred_label, "score": 1.0, "segment": [round(float(prev_pred_idx), 4), round(float(step_idx), 4)]})
                prev_pred_idx = step_idx
                prev_pred_label = dict_value[0]['pred_label']

            if prev_gt_label != dict_value[0]['gt_label'] or step_idx == 99:  # or ret_idx == len(ret):
                gt_action_list.append({"label": prev_gt_label,
                                       # "score": 1.0,
                                       "segment": [round(float(prev_gt_idx), 4), round(float(step_idx), 4)]})
                prev_gt_idx = step_idx
                prev_gt_label = dict_value[0]['gt_label']

    pred_action_list.append({"label": prev_pred_label, "score": 1.0,
                             "segment": [round(float(prev_pred_idx), 4), round(float(step_idx), 4)]})
    gt_action_list.append({"label": prev_gt_label,
                           # "score": 1.0,
                           "segment": [round(float(prev_gt_idx), 4), round(float(step_idx), 4)]})
    detection_dict_gt[vid_flag] = {"subset": "test", "annotations": gt_action_list}
    detection_dict_pred[vid_flag] = pred_action_list

    new_gtad_prediction = {"version": "Babel", "results": detection_dict_pred, "external_data": {}}
    new_gtad_gt = {"database": detection_dict_gt}
    os.makedirs('../../../gtad/output/default/', exist_ok=True)
    detection_dict_pred_path = '../../../gtad/output/default/babel_detection_result_pred_retrieval.json'
    detection_dict_gt_path = '../../../gtad/output/default/babel_detection_result_gt_retrieval.json'
    with open(detection_dict_gt_path, "w") as out:
        json.dump(new_gtad_gt, out)
        print('Save to {}'.format(detection_dict_gt_path))
    with open(detection_dict_pred_path, "w") as out:
        json.dump(new_gtad_prediction, out)
        print('Save to {}'.format(detection_dict_pred_path))