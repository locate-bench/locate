#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 achandrasekaran <arjun.chandrasekaran@tuebingen.mpg.de>
#
# Distributed under terms of the MIT license.

"""
Data-related classes and functions
"""

import os, sys, random, pdb, copy
import os.path as osp
import numpy as np
import json, pickle
from collections import defaultdict, Counter, OrderedDict


def generate_json_from_SMPL(
        data_root='../../data/babel',
        dataset='babel',
        AR_feature_root='../../../data/babel/babel_smpl_sk_Nv_100_csz_8_joint_groundcontact.npy',
        babel_label_path='../../../babel_tools/babel_v1.0_release',
        npz_prefix='../../../Dataset/amass_unzip_smpl_jpos',
        amass_unzip_prefix='../../../Dataset/amass_unzip',
        num_chunks_per_sequence=100,
        feat_array_last_shape=600,
        make_stat=True,
        save_json_and_data=True,
        with_groundcontact=True,
        thres=0.9,
        ):
    features_root = os.path.join(data_root, 'i3d_feats')
    os.makedirs(features_root, exist_ok=True)
    vid_data_file = os.path.join(data_root, dataset + "_action.json")

    # mapping_file
    mapping_file = os.path.join(data_root, "action_mapping.txt")
    # version 1
    idx2als = {0: ['walk'],
               1: ['hit', 'punch'],
               2: ['kick'],  # Really short action segments?
               3: ['run', 'jog'],
               4: ['jump', 'hop', 'leap'],
               5: ['throw'],
               6: ['catch'],
               7: ['step'],
               8: ['greet'],  # Large diversity expected
               9: ['dance'],
               10: ['stretch', 'yoga', 'exercise / training'],  # Large diversity expected
               11: ['turn', 'spin'],  # Is this distinct enough from take / pick something up? Yeah, I think so.
               12: ['bend'],
               13: ['stand'],  # Large diversity expected
               14: ['sit'],
               15: ['kneel'],
               16: ['place something'],
               17: ['grasp object'],
               18: ['take/pick something up', 'lift something'],
               19: ['scratch', 'touching face', 'touching body parts'],
               }
    idx2al = {}
    for k, v in idx2als.items():
        idx2al[k] = ' or '.join(v)
    print('idx2al: ', idx2al)
    al2idx = {idx2al[k]: k for k in idx2al}

    top_60_mapping_str = idx2al
    print('top_60_mapping: ', top_60_mapping_str)
    if AR_feature_root.endswith('.npy'):
        spl_X = np.load(AR_feature_root)  # spl_X.shape:  (808900, 3, 8, 25, 1)
        print('spl_X.shape: ', spl_X.shape)  # spl_X.shape:  (8808, 100, 8, 96)
        if with_groundcontact:
            if thres:
                print('spl_X[0, 0, 0, 24*3:]: ', spl_X[0, :3, 0, 24*3:])
                spl_X[:, :, :, 24*3:] = spl_X[:, :, :, 24*3:] > thres
                print('spl_X[0, 0, 0, 24*3:]: ', spl_X[0, :3, 0, 24 * 3:])
            spl_X = spl_X.reshape(spl_X.shape[0], spl_X.shape[1], -1)
        else:
            spl_X = spl_X[:, :, :, :24 * 3].reshape(spl_X.shape[0], spl_X.shape[1], -1)
        print('spl_X.shape: ', spl_X.shape)  # spl_X.shape:  (8808, 100, 768)
    else:
        with open(os.path.join(AR_feature_root, '{}_feat.pkl'.format('epoch1_test')), 'rb') as f:
            feat_dict = pickle.load(f)
        spl_X = feat_dict['feat']  # feat_array.shape: (808900, 256)
        spl_X = spl_X.reshape(int(spl_X.shape[0] // num_chunks_per_sequence), num_chunks_per_sequence, spl_X.shape[1])
    print('spl_X.shape: ', spl_X.shape)  # spl_X.shape:  (27838, 256)

    train_json = os.path.join(babel_label_path, 'train.json')  # train_extra.json
    val_json = os.path.join(babel_label_path, 'val.json')  # val_extra.json
    splits = {'train': [], 'val': []}
    sample_paths = []
    d = {'X': [], 'Psi': [], 'fps': [], 'ft_type': 'pos', 'url': [], 'feat_path': [], 'vids': [], 'amass': []}

    with open(train_json, 'r') as f:
        train_label = json.load(f)
    for i, (k, v) in enumerate(train_label.items()):
        amass_feats = '/'.join(v['feat_p'].split('/')[1:])
        npz_file = np.load(os.path.join(npz_prefix, amass_feats.replace('.npz', '.npy')))
        amass_npz = np.load(os.path.join(amass_unzip_prefix, amass_feats))
        d['X'].append(copy.deepcopy(npz_file.reshape(npz_file.shape[0], -1)))
        sample_paths.append(os.path.join(amass_unzip_prefix, amass_feats))
        try:
            labels = v['frame_ann']['labels']
            Psi = []
            for label in labels:
                for each_act_cat in label['act_cat']:
                    Psi.append({'ts': label['start_t'] / v['dur'], 'te': label['end_t'] / v['dur'], 'c': each_act_cat})
            d['Psi'].append(Psi)
        except:
            labels = v['seq_ann']['labels']
            Psi = []
            for label in labels:
                for each_act_cat in label['act_cat']:
                    Psi.append({'ts': 0., 'te': v['dur'], 'c': each_act_cat})
            d['Psi'].append(Psi)
        d['fps'].append(amass_npz['mocap_framerate'])
        d['url'].append(v['url'])
        d['vids'].append(v['url'])
        splits['train'].append(v['url'])
        d['feat_path'].append(amass_feats)
        if i % 1000 == 0:
            print('Processing {} / {}'.format(i, len(train_label)))
            print([k for k in amass_npz.files], amass_npz['mocap_framerate'])

    print('Val Set')
    with open(val_json, 'r') as f:
        val_label = json.load(f)
    for i, (k, v) in enumerate(val_label.items()):
        amass_feats = '/'.join(v['feat_p'].split('/')[1:])
        npz_file = np.load(os.path.join(npz_prefix, amass_feats.replace('.npz', '.npy')))
        amass_npz = np.load(os.path.join(amass_unzip_prefix, amass_feats))
        d['X'].append(copy.deepcopy(npz_file.reshape(npz_file.shape[0], -1)))
        sample_paths.append(os.path.join(amass_unzip_prefix, amass_feats))
        try:
            labels = v['frame_ann']['labels']
            Psi = []
            for label in labels:
                for each_act_cat in label['act_cat']:
                    Psi.append({'ts': label['start_t'] / v['dur'], 'te': label['end_t'] / v['dur'], 'c': each_act_cat})
            d['Psi'].append(Psi)
        except:
            labels = v['seq_ann']['labels']
            Psi = []
            for label in labels:
                for each_act_cat in label['act_cat']:
                    Psi.append({'ts': 0., 'te': v['dur'] / v['dur'], 'c': each_act_cat})
            d['Psi'].append(Psi)
        d['fps'].append(amass_npz['mocap_framerate'])
        d['url'].append(v['url'])
        d['vids'].append(v['url'])
        splits['val'].append(v['url'])
        d['feat_path'].append(amass_feats)
        if i % 1000 == 0:
            print('Processing {} / {}'.format(i, len(val_label)))
            print([k for k in amass_npz.files], amass_npz['mocap_framerate'])

    d['ann_idx'] = list(range(len(d['Psi'])))
    data = d

    Psi = data['Psi']
    spl_Psi_c = []
    spl_Psi_ts = []
    spl_Psi_te = []
    for each_psi in Psi:
        each_psi_c = []
        each_psi_ts = []
        each_psi_te = []
        for item in each_psi:
            if item['c'] in ['run', 'jog']:
                action_class = 'run or jog'
            elif item['c'] in ['jump', 'hop', 'leap']:
                action_class = 'jump or hop or leap'
            elif item['c'] in ['stretch', 'yoga', 'exercise / training']:
                action_class = 'stretch or yoga or exercise / training'
            elif item['c'] in ['turn', 'spin']:
                action_class = 'turn or spin'
            elif item['c'] in ['take/pick something up', 'lift something']:
                action_class = 'take/pick something up or lift something'
            elif item['c'] in ['scratch', 'touching face', 'touching body parts']:
                action_class = 'scratch or touching face or touching body parts'
            elif item['c'] in ['hit', 'punch']:
                action_class = 'hit or punch'
            else:
                action_class = item['c']
            if action_class not in al2idx:
                continue
            each_psi_c.append(al2idx[action_class])
            each_psi_ts.append(item['ts'] * num_chunks_per_sequence)
            each_psi_te.append(item['te'] * num_chunks_per_sequence)
        spl_Psi_c.append(each_psi_c)
        spl_Psi_ts.append(each_psi_ts)
        spl_Psi_te.append(each_psi_te)
    spl_vids = data['vids']
    ann_idx = data['ann_idx']

    print('spl_vids.shape: ', len(spl_vids))  # spl_vids.shape:  (27838,)
    print('ann_idx.shape: ', len(ann_idx))
    spl_lidxs = defaultdict(list)
    for idx, vid in enumerate(data['vids']):
        vid_spl = [spl for spl in splits if vid in splits[spl]][0]
        spl_lidxs[vid_spl].append(idx)
    spl_aidxs = {spl: np.array(spl_lidxs[spl]) for spl in spl_lidxs}

    for spl in spl_aidxs:
        print('BABEL dataset # {0} = {1}'.format(spl, len(spl_aidxs[spl])))
    spl_idxs = spl_aidxs

    num_total_videos = 0
    num_ignored_videos = 0
    num_npy_videos = 0
    subsets = ['train', 'val']
    action_json = OrderedDict()
    idx_video = 0
    num_queries_list = []  # number of action queries, ie detection slot. This is the maximal number of actions that can be detected in a video.
    class_seg_num = {'train': {}, 'val': {}, 'test': {}}
    for subset in subsets:
        idxs = np.array(spl_idxs[subset])
        print()
        for i, (video_name, feat_array, start_array, end_array, action_array, ann_idx_array) \
                in enumerate(zip(spl_vids, spl_X, spl_Psi_ts, spl_Psi_te, spl_Psi_c, ann_idx)):
            if ann_idx_array not in idxs:
                continue
            num_total_videos += 1
            if idx_video % 1000 == 0:
                print('Getting video %d / %d' % (idx_video, len(spl_vids)), flush=True)
            video_name = video_name.replace('https://crichton.is.tue.mpg.de/', '').replace('/', '-')
            video_name = video_name + '@' + '{0:07d}'.format(ann_idx_array)
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
            for (start_frame, end_frame, act_cat) in zip(start_array, end_array, action_array):
                action_list.append(
                    [int(act_cat), round(start_frame * 1., 1), min(round(feat_array.shape[0] - 1.0, 1), round(end_frame * 1., 1))]
                )
                action_list = action_list[:20]
                if idx2al[act_cat] not in class_seg_num[subset]:
                    class_seg_num[subset][idx2al[act_cat]] = 1
                else:
                    class_seg_num[subset][idx2al[act_cat]] += 1

            if len(action_list) > 0:
                num_queries_list.append(len(action_list))
                label = {"duration": round(num_chunks_per_sequence - 1.0, 2), "subset": label_subset, "actions": action_list}
                action_json['{}'.format(video_name)] = label
                if feat_array.shape[-1] < feat_array_last_shape:
                    padding_feat_array = np.concatenate((feat_array, np.zeros((feat_array.shape[0], feat_array_last_shape - feat_array.shape[1]))), axis=1)
                else:
                    padding_feat_array = feat_array
                if save_json_and_data:
                    np.save(os.path.join(features_root, video_name+'.npy'), padding_feat_array[:, None, None, :])  #  (time_step, H, W, feat_dim)

                num_npy_videos += 1
            else:
                num_ignored_videos += 1
            idx_video = idx_video + 1
    print('class_seg_num: ', class_seg_num)
    if make_stat:
        num_queries_counter = Counter(num_queries_list)
        print('num_queries_counter: ', num_queries_counter)

        num_queries = max(num_queries_list)
        mean_num_queries = np.mean(num_queries_list)
        median_num_queries = np.median(num_queries_list)
        std_num_queries = np.std(num_queries_list)
        min_num_queries = np.min(num_queries_list)
        max_num_queries = np.max(num_queries_list)
        print('num_queries: {}, mean_num_queries: {}, median_num_queries: {}, std_num_queries: {}, min_num_queries: {}, max_num_queries: {}'
              .format(num_queries, mean_num_queries, median_num_queries, std_num_queries, min_num_queries, max_num_queries))
    if save_json_and_data:
        with open(vid_data_file, 'w') as f:
            json.dump(action_json, f)
        print('Json file saved to {}'.format(vid_data_file))
        with open(mapping_file, 'w') as f:
            for k, v in top_60_mapping_str.items():
                f.write("{} {}\n".format(k, v.replace(' ', '_')))
        print('Mapping file saved to {}'.format(mapping_file))
    print('Num ignored videos: {} / {}, '.format(num_ignored_videos, num_total_videos),
          'Num saved npy videos: {} / {}'.format(num_npy_videos, num_total_videos))

if __name__ == "__main__":
    generate_json_from_SMPL()