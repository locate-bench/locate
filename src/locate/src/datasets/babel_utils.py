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
from itertools import combinations
from collections import defaultdict, Counter, OrderedDict
from scipy.interpolate import interp1d

from smplx import SMPL

import torch
from torch.utils.data import Dataset, DataLoader


class BABELDetection:
    '''Detection (temporal action localization) dataset from BABEL.

      Members:
          self.vids (list of str): URLs of the rendered mocap sequences.
          self.X (np.array): $$x_{0:T}$$ = mocap sequence with $$T$$ frames.
          self.Psi (list of dicts): $$\Psi_N = \{ t_s, t_e, c \}^N$$ which are
              the $$N$$ GT detections (segments).
              $$t_s, t_e, c$$ = start time, end time, GT action category.
    '''
    def __init__(self,
                 dataset_path='/home/jksun/Dataset/amass_dataset/amass_act.min.json',
                 ft_type='', # 'pos',
                 skip_cats=[],
                 toy_size=None
                ):
        '''Loads BABEL for temporal localization. Classes are action
        categories. All categories are loaded by default.

        Args:
            dataset_path (str): Path to BABEL dataset JSON.
            ft_type (str): 'pos' for joint positions, 'aa' for axis angles.
            skip_cats (list of str): Action categories to exclude from dataset.
            toy_size (int): Number of mocap sequences to return as data.
        '''
        # Input arguments
        self.toy_size = np.inf if toy_size is None else toy_size
        self.skip_cats = [act.strip().lower() for act in skip_cats]
        self.ft_type = ft_type

        # Map: action categories <-- raw string action labels
        self.rawl_to_act = {}
        # with open('../semi-param-motion/analysis/fg_coarse_map.json') as infile:
        with open('prep_data/fg_coarse_map.json', 'r') as infile:
            self.rawl_to_act = json.load(infile)

        # 'Output' members
        self.X, self.Psi, self.vids, self.fps = [], [], [], []

        if os.path.splitext(dataset_path)[-1] == '.pkl':
            with open(dataset_path, 'rb') as infile:
                babel = pickle.load(infile)  # len = 43888
        else: # json
            with open(dataset_path, 'r') as infile:
                babel = json.load(infile)
        for i, vid in enumerate(babel):
            amass_seq_path = babel[vid]['amass_feats_path']
            amass_seq_path = amass_seq_path.replace('/ps/project/datasets/AMASS/',
                                                                 '/home/jksun/Dataset/amass_dataset/')\
                .replace('BMLrub', 'BioMotionLab_NTroje') \
                .replace('DFaust', 'DFaust_67')
            babel[vid]['amass_feats_path'] = amass_seq_path

        self._get_vids_with_label(babel)

        self.Psi = np.array(self.Psi)
        self.vids = np.array(self.vids)
        self.fps = np.array(self.fps)
        self.X = np.array(self.X)
        os.makedirs('data', exist_ok=True)

    def _get_seq_from_amass(self, amass_seq_path):
        """From: https://github.com/nghorbani/amass/blob/master/notebooks/01-AMASS_Visualization.ipynb
        For a particular frame, say, fId=0,
        root_orient = bdata['poses'][fId:fId+1, :3])
        pose_body = bdata['poses'][fId:fId+1, 3:66])
        pose_hand = bdata['poses'][fId:fId+1, 66:]
        """
        bdata = np.load(amass_seq_path)
        # print(bdata['poses'].shape)
        amass_feats = {'fps': bdata['mocap_framerate']}
        # All frames in sequence, [root orientation, pose of body] (no hands)
        body_feats = bdata['poses']
        transl = bdata['trans']
        if 'pos' == self.ft_type:
            # Read stored files from disk
            jpos_fpath = amass_seq_path.split('/ps/project/datasets/AMASS/')[-1]
            jpos_full_fpath = os.path.join('/ps/project/conditional_action_gen/amass/joint_pos', jpos_fpath)
            if not os.path.isfile(jpos_full_fpath):
                dirpath = os.path.dirname(jpos_full_fpath)
                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)
                body_feats, feats = self.get_pos_from_aa(body_feats, transl)
                np.savez(os.path.join(dirpath, os.path.basename(jpos_full_fpath)), \
                                        joint_pos=body_feats, \
                                        joint_pos_w_constr=feats, \
                                        allow_pickle=True)
                print('Created: ', jpos_full_fpath)
            else:
                bpos_data = np.load(jpos_full_fpath)
                body_feats = bpos_data['joint_pos']
                # Ignoring constraints since it's not implemented yet
        amass_feats['seq'] = np.hstack((transl, body_feats))
        return amass_feats

    def _get_stage_1_data(self, vid, vid_ann):
        '''Include Stage 1 sequence in dataset if
        both subjects agree that it has just one action.

        Args:
            vid (str): URL of the video. Pass this to self._append_feats()
            vid_ann (dict): All metadata and annotations for given sequence.
        '''
        # Identify Stage 1 task
        assert 1 == len(vid_ann['stage_1_data'].keys())
        task = list(vid_ann['stage_1_data'].keys())[0]
        # Annotations for this seq.
        anns = vid_ann['stage_1_data'][task]
        if len(anns) < 1:
            return

        # Get labels, multiple-action info from Stage 1
        proc_collated_labels, mul_act = [], True

        # In Pilot 1 each annotation = list of actions <str>
        if 'p1' == task:
            # pdb.set_trace()
            mul_act = True if max(len(anns[0]), len(anns[1])) > 1 else False
            proc_collated_labels = anns
        # In all other Stage 1 tasks, each annotation = <str>
        else:
            mul_act = anns[0]['multiple_actions'] or anns[1]['multiple_actions']
            collated_labels = [[ann['action_labels']] for ann in anns]
            proc_collated_labels = [[ann['pre_processed']] for ann in anns]

        if mul_act is True:
            return

        # Load feats
        feats = self._get_seq_from_amass(vid_ann['amass_feats_path'])
        # Loop over each Turker's annotation
        for labels in proc_collated_labels:
            ls = [l.lower().strip() for l in labels if l.strip() != '']
            for l in ls:
                if l not in self.rawl_to_act:
                    print('{0} is not mapped to an action category'.format(l))
                    continue
                for act_cat in self.rawl_to_act[l]:
                    Psi = [{'ts': 0,
                            'te': feats['seq'].shape[0],
                            'c': act_cat }]
                    self.vids.append(vid)
                    self.fps.append(feats['fps'])
                    self.X.append(feats['seq'])
                    self.Psi.append(Psi)

    def _get_proc_stage_2_label(self, seg_dict):
        '''
        '''
        # Get label from dict. Handle
        proc_label = None
        # print('seg_dict: ', seg_dict)
        if 'pre_processed' in seg_dict:
            proc_label = seg_dict['pre_processed']
            del seg_dict['pre_processed']
            if proc_label == 'touching hand':
                proc_label = 'touching hands together'
            elif proc_label == 'rotates hand inwards':
                proc_label = 'rotates hands inwards'
            elif proc_label == 'rotate shoulder forwards':
                proc_label = 'rotates shoulders forwards'
            elif proc_label == 'rotates hand outward':
                proc_label = 'rotates hands outwards'
            elif proc_label == 'rotate shoulder backwards':
                proc_label = 'rotates shoulders backwards'
            elif proc_label == 'kick foot backwards forwards':
                proc_label = 'kicks both feet backwards then forwards'
            elif proc_label == 'kick foot backwards':
                proc_label = 'kicks both feet backwards'
            elif proc_label == 'low right hand':
                proc_label = 'lower right hand to side'
            elif proc_label == 'low arm':
                proc_label = 'lower arms to side'
            elif proc_label == 'place hand floor':
                proc_label = 'place hands on floor'
            elif proc_label == 'loosening grip hand':
                proc_label = 'loosening grips of the hands'
            elif proc_label == 'shuffle foot':
                proc_label = 'shuffle feet'
            elif proc_label == 'shoulder':
                proc_label = 'move shoulders'
            elif proc_label == 'hop right leg':
                proc_label = 'hop on right leg'
            elif proc_label == 'hop left leg':
                proc_label = 'hop on left leg'
            elif proc_label == 'step forward left leg':
                proc_label = 'steps forward left leg'
            elif proc_label == 'pivot body degree':
                proc_label = 'pivots body 180 degrees'
            elif proc_label == 'push large object hand':
                proc_label = 'pushes against large object using both hands'
            elif proc_label == 'touching face right hand':
                proc_label = 'touching face with right hand'
            elif proc_label == 'jog degree right':
                proc_label = 'jog 180 degrees right'
            elif proc_label == 'climb step':
                proc_label = 'climb in the steps'
        # assert 1 == len(seg_dict.keys()), 'len(seg_dict.keys()): {}'.format(seg_dict.keys())
        label = list(seg_dict.keys())[0]
        if proc_label is None:
            proc_label = label
        # print(proc_label, label)
        return proc_label, label

    def get_stage_2_data(self, vid, vid_ann):
        '''
        '''
        # amass_seq_path = vid_ann['amass_feats_path'].replace('/ps/project/datasets/AMASS/', '/home/jksun/Dataset/amass_dataset/')
        # amass_seq_path = amass_seq_path.replace('BMLrub', 'BioMotionLab_NTroje')
        # amass_seq_path = amass_seq_path.replace('DFaust', 'DFaust_67')
        feats = self._get_seq_from_amass(vid_ann['amass_feats_path'])
        self.vids.append(vid)
        self.fps.append(feats['fps'])
        self.X.append(feats['seq'])
        Psi = []

        # Handle "error" tasks while parsing task name
        task = [k for k in vid_ann['stage_2_data'] if 'error' not in k][0]
        # List of segments in task
        segs = vid_ann['stage_2_data'][task]
        for seg_dict in segs:
            proc_label, label = self._get_proc_stage_2_label(seg_dict)
            if proc_label is None or proc_label == '':
                continue
            for act_cat in self.rawl_to_act[proc_label]:
                Psi.append({'ts': int(seg_dict[label][0]*feats['fps']),
                            'te': int(seg_dict[label][1]*feats['fps']),
                            'c': act_cat
                            })
        self.Psi.append(Psi)

    def _get_vids_with_label(self, babel):
        '''For each sequence, store relevant data in appropriate data structure.

        Args:
            babel (dataset JSON): Annotations for each mocap sequence.
                    [{'vid': '<url>', 'labels': <Stage 1 labels>}, ...]
        '''
        for i, vid in enumerate(babel):

            # Terminate after desired dataset size is reached
            if  self.toy_size*3 <= len(self.vids):
                return None

            # Load Babel data -- fine-grained labels for entire seq.
            if len(babel[vid]['stage_2_data']) > 0:
                self.get_stage_2_data(vid, babel[vid])
            if len(babel[vid]['stage_1_data']) > 0:
                self._get_stage_1_data(vid, babel[vid])

            # Display progress
            if 0 == i%500:
                print('Done with {0}/{1} sequences.'.format(i+1, len(babel)))
                break

        return None


def _get_frame_feats(frames, ft_type=None):
    '''Return a feature for each input frame

    Args:
        frames (float) (T, in_feat_sz): Seq. whose feats we need.
            Assumption: Each frame is the SMPLH pose.
        ft_type (str): Type of frame feature. Default=None.
            None: Returns input frames as is.
            'smpl': The feature of a frame is its SMPL pose parameters.

    Returns:
        frame_feats (float) (T, feat_size): Features for all frames.
    '''
    if ft_type is None:
        return frames
    elif 'smpl' == ft_type:
        raise NotImplementedError
        # frame_feats = frames[:, 0:63]
        # return frame_feats
    else:
        raise NotImplementedError


def _get_snippet_feat(frames,
                     sigma=10,
                     pool_type='avg',
                     snip_feat_type='smpl',
                     exact_sigma=False):
    '''Return the feature for a given snippet. Described in Sec. 3.1.

    Args:
        frames (float) (T, in_feat_sz): Seq. whose snippet feats we need.
        sigma (int): Number of frames in one snippet.
        pool_type (str): Method to aggregate the sigma frame feats in snippet.
            'avg': Compute mean of the sigma frame features.
        snip_feat_type (str): Type of frame feature.
            'smpl': The feature of a frame is its SMPL pose parameters.
        ignore_extra (bool): False - Extra frames are included as separate
            snippet with < sigma frames. True - error if frames are not
            perfectly divisible into some integer (T_S) snippets.

    Returns:
        snip_ft (float) (N/sigma, feat_size): Snippets' features.
    '''
    # Get features for each frame in the sequence
    fr_ft = _get_frame_feats(frames)

    # Reshape per-frame feats into T_S snippets
    T, in_ft_sz = np.shape(frames)
    n_extra_fr = T % sigma

    # Aggregate feats for the perfectly divisible snippets and extra snippet
    snips_ft = None
    if 0 == n_extra_fr:
        fr_ft = fr_ft.reshape((-1, sigma, in_ft_sz))
        if 'avg' == pool_type:
            snips_ft = np.mean(fr_ft, axis=1)
    else:
        if exact_sigma:
            assert ValueError('{0} frames cannot be exactly divided into \
                              snippets of length {1} each!'.format(sigma))
        else:
            # Get all the full snippets
            ex_snips_ft = fr_ft[:-n_extra_fr, :].reshape((-1, sigma, in_ft_sz))
            if 'avg' == pool_type:
                snips_ft = np.mean(ex_snips_ft, axis=1)
                extra_snip_ft = np.mean(fr_ft[np.newaxis, -n_extra_fr:, :], axis=1)
                snips_ft = np.concatenate((snips_ft, extra_snip_ft), axis=0)

    return snips_ft


def _get_L_snips(snips_ft, L):
    '''Return L snippets.

    Args:
        L (int): Fixed number of snippets per mocap sequence.

    Returns:
        snips_ft (np.array L*N*ft_sz): Snippet features for N sequences, each
            having features of size ft_sz.
    '''
    if len(snips_ft) == L:
        return snips_ft
    else:
        W, M = snips_ft.shape
        x = np.arange(W)
        itp_A2 = interp1d(x, snips_ft.T, kind='nearest')
        xi = np.linspace(0, W-1, L)
        new_snips_ft = itp_A2(xi)
        return new_snips_ft.T


def _vids_to_snips(seqs, n_snips_per_seq=100):
    '''Return $$L$$ snippets for each of the untrimmed input sequences.

    Args:
        seqs (list of motion features): Dataset of mocap sequences.
        n_snip_per_seq (int): Number of snippets per seq.
    '''
    snips = []
    for x in seqs:
        snips_ft = _get_snippet_feat(x)
        snips_ft = _get_L_snips(snips_ft, L=n_snips_per_seq)
        snips.append(snips_ft)
    return snips


def process_raw_data(data):
    '''Process data in the following ways:
    1. Convert original mocap sequence features from original frame-rate to
    30fps.
    2. Convert starting and ending frame indices from original frame idxs to
    corresponding frame idxs in the new 30fps sequence.
    3. Ensure that all Psi labels (segments) are of the same length across
    samples.
    4. Switch the string labels with class indices

    Returns:
        X (float, np.array): Mocap sequence features in original frame-rate.
        vids (str, np.array): URL for each sequence.
        - Below 3 arrays are of length max_Psi (max. # segments in a sequence)
        Psi_ts (int, np.array): Array of starting frame of each segment.
        Psi_te (int, np.array, ): Array of ending frame of each segment.
        Psi_c (int, np.array): Array of action label index for each segment.
    '''
    # Load action label --> idx mapping
    al_to_idx_fpath = '/home/jksun/Program/gtad/packages/activitygraph_transformer/data/babel_annotations/action_label_to_idx.json'
    if osp.exists(al_to_idx_fpath):
        with open(al_to_idx_fpath, 'r') as infile:
            al_to_idx = json.load(infile)
    else:
        set_als = sorted(list(set([psi['c'] for Psi in data['Psi'] for psi in Psi])))
        al_to_idx = {al: alidx for alidx, al in enumerate(set_als)}
        with open(al_to_idx_fpath, 'w') as outfile:
            json.dump(al_to_idx, outfile)

        babel_class_path = "/home/jksun/Program/gtad/packages/activitygraph_transformer/data/babel_annotations/babel_class.json"
        babel_class = {0: 'None'}
        for i, (k, v) in enumerate(al_to_idx.items()):
            babel_class[v + 1] = k
        # print('babel_class: ', )
        # pprint.pprint(babel_class)
        with open(babel_class_path, 'w') as f:
            json.dump(babel_class, f)

    # Make sure that # labels (Psi) is the same for all samples
    max_Psi = max([len(Psi) for Psi in data['Psi']])

    # Num samples
    new_data = {'X': [],\
                'Psi_c': -1+np.zeros((len(data['vids']), max_Psi), dtype=int),
                'Psi_ts': -1+np.zeros((len(data['vids']), max_Psi), dtype=int),
                'Psi_te': -1+np.zeros((len(data['vids']), max_Psi), dtype=int),
                'vids': data['vids'],
                'skip_cats': data['skip_cats'],
                'ft_type': data['ft_type']
                }

    for idx, x in enumerate(data['X']):
        # Subsample to 30fps
        step = int(data['fps'][idx]/30.0)
        new_data['X'].append(x[::step])

        # Adjust start and end frames to 30fps equivalent frame numbers.
        for psi_idx, psi in enumerate(data['Psi'][idx]):
            new_data['Psi_ts'][idx][psi_idx] = int(psi['ts']/step)
            new_data['Psi_te'][idx][psi_idx] = int(psi['te']/step)
            new_data['Psi_c'][idx][psi_idx] = al_to_idx[psi['c']]

    new_data['vids'] = np.array(new_data['vids'])
    return new_data


def remove_short_seqs(data):
    '''
    '''
    print('Removing sequences that are < 1s in duration.')
    fil_data = {'X': [], 'Psi': [], 'vids': [], 'fps': [], \
                'skip_cats': data['skip_cats'], 'ft_type': data['ft_type'] }
    for i, x in enumerate(data['X']):
        if (np.shape(x)[0]/data['fps'][i]) >= 1.0:
            for k in ['X', 'Psi', 'vids', 'fps']:
                fil_data[k].append(data[k][i])
    print('# Sequences = ', len(fil_data['vids']))
    return fil_data


def store_and_load_babel():
    '''Store if doesn't already exist, and load BABEL GT data.

        Returns:
            data = {'X': d.X, 'Psi': d.Psi, 'vids': d.vids, 'fps': d.fps,
               'skip_cats': d.skip_cats, 'ft_type': d.ft_type}
    '''
    data_fpath = '/home/jksun/Program/gtad/data/babel_feature/babel_detection_obj.pkl'
    if os.path.exists(data_fpath):
        with open(data_fpath, 'rb') as infile:
            data = pickle.load(infile)
    else:
        d = BABELDetection()
        data = {'X': d.X, 'Psi': d.Psi, 'vids': d.vids, 'fps': d.fps,
               'skip_cats': d.skip_cats, 'ft_type': d.ft_type}
        with open(data_fpath, 'wb') as outfile:
            pickle.dump(data, outfile)
    print('Successfully loaded BABEL dataset!')
    print('Total # sequences = ', len(data['vids']))

    # Filter out sequences if < 1s
    fil_data = remove_short_seqs(data)

    # Make sure that the output seqs are at 30fps, |Psi| = k across samples.
    processed_data = process_raw_data(fil_data)
    return processed_data


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
    with open(osp.join(split_path, 'babel_splits.json')) as infile:
        splits = json.load(infile)

    # Get indices for split samples
    for idx, vid in enumerate(data['vids']):
        vid_spl = [spl for spl in splits if vid in splits[spl]][0]
        spl_lidxs[vid_spl].append(idx)
    spl_aidxs = {spl: np.array(spl_lidxs[spl]) for spl in spl_lidxs}

    for spl in spl_aidxs:
        print('BABEL dataset # {0} = {1}'.format(spl, len(spl_aidxs[spl])))
    return spl_aidxs


def get_pt_dataloaders(batch_size, shuffle=True):
    '''1. Load BABEL GT data of the format:
          data = {'X': d.X, 'Psi': d.Psi, 'vids': d.vids, 'fps': d.fps,
                 'skip_cats': d.skip_cats, 'ft_type': d.ft_type}
      2. Get snippet features.
      3. Then, split this into appropriate train, val, test splits.
      4. Create dataloaders for each split.
    '''
    # Get raw data from BABEL
    data = store_and_load_babel()

    # Convert raw sequences into snippets
    data['snips'] = np.array(_vids_to_snips(data['X']))
    print('Shape of snippets = ', np.shape(data['snips']))

    # Split data into train, val, test
    spl_idxs = get_split_idxs(data)

    # Get loaders for each split
    dsets, dloaders = {}, {}
    for spl in spl_idxs:
        idxs = spl_idxs[spl]
        dsets[spl] = BABELDetectionDataset(spl_X=data['snips'][idxs],
                                        spl_Psi_ts=data['Psi_ts'][idxs],
                                        spl_Psi_te=data['Psi_te'][idxs],
                                        spl_Psi_c=data['Psi_c'][idxs],
                                        spl_vids=data['vids'][idxs])
        print('Created {0} dataset'.format(spl))
        dloaders[spl] = DataLoader(dsets[spl], batch_size=batch_size,
                                        shuffle=shuffle, drop_last=True)
        print('Created {0} dataloader'.format(spl))
    return dsets, dloaders


class BABELDetectionDataset(Dataset):
    def __init__(self, spl_X, spl_Psi_ts, spl_Psi_te, spl_Psi_c, spl_vids):
        ''' Initialize Pytorch Dataset

        Args:
            spl_X (float, np.array): Features of samples belonging to split.
            spl_Psi_ts (int, np.array): Starting frames for all segments.
            spl_Psi_te (int, np.array): Ending frames for all segments.
            spl_Psi_c (str, np.array): Action labels for all segments.
            spl_vids (list of URLs): URLs in split.
        '''
        self.spl_X = spl_X
        self.spl_Psi_ts = spl_Psi_ts
        self.spl_Psi_te = spl_Psi_te
        self.spl_Psi_c = spl_Psi_c
        self.spl_vids = spl_vids
        # print('self.spl_X.shape: {}, self.spl_Psi_ts.shape: {}, self.spl_Psi_te.shape: {}, self.spl_Psi_c.shape: {}, self.spl_vids.shape: {}'
        #       .format(self.spl_X.shape, self.spl_Psi_ts.shape, self.spl_Psi_te.shape, self.spl_Psi_c.shape, len(self.spl_vids)))
        # self.spl_X.shape: (1573, 100, 69), self.spl_Psi_ts.shape: (1573, 95), self.spl_Psi_te.shape: (1573, 95),
        # self.spl_Psi_c.shape: (1573, 95), self.spl_vids.shape: 1573
        # print(self.spl_Psi_ts[0], self.spl_Psi_te[0], self.spl_Psi_c[0], self.spl_vids[0])
    def __len__(self):
        ''' PyTorch Dataset - overloaded function. Return length of split.'''
        return len(self.spl_vids)

    def __getitem__(self, idx):
        ''' Pytorch Dataset - overloaded function. Return 1 sample.
        '''
        # Sample = (mocap features, temporal labels, url of video)
        return self.spl_X[idx], self.spl_Psi_ts[idx], self.spl_Psi_te[idx],\
               self.spl_Psi_c[idx], self.spl_vids[idx]

        # Sample = (mocap features for L frames, labels for L frames, vids)
        # pdb.set_trace()
        # s_label = -1 + np.zeros(self.spl_X[idx].shape[0])
        # for psi_i, st_f in enumerate(self.spl_Psi_ts[idx]):
        #     # TODO: Nice way to handle simultaneous actions.
        #     s_label[(st_f): self.spl_Psi_te[idx][psi_i]] = self.spl_Psi_c[idx][psi_i]


def ref_prep_seq(seq, data_format, inc_trans=True, inc_root=True):
    ''' Refactored code for `preprocess_motion_seq(motion).
    Tested successfully with (some) AMASS data.
    :data_format <str>: 'smpl_js' stores (not returned yet) translation,
                                root orientation and body joints except hands.
                        'smplh_js' stores (not returned yet) translation + all
                                                                SMPLH joints.
    :inc_trans <bool>: If True, keep translation as first 3 dims. Else, remove.
    :inc_root <bool>: If True, keep root orientation. If translation is
                    included, then root dims are [3:6] dims.
                    Else root are [0:3] dims.
    '''
    if data_format == 'smpl_js':
        seq = seq[:, :, :69]  # Trans + root + 21*3 body joints
    elif data_format == 'smplh_js':
        seq = seq[:, :, :]  # Keep full sequence
    # Above seq. corresponds to inc_trans=True, inc_root=True

    if False == inc_trans and True == inc_root:
        seq = seq[:, :, 3:]
    elif False == inc_trans and False == inc_root:
        seq = seq[:, :, 6:]
    # NOTE: Not handling case where inc_trans=True and inc_root=False

    return seq


def thumos_name_dicts():
    src_dict = {'validation': 1, 'test': 2}

    return src_dict


def getVideoId(video_name):
    src_name = video_name.split('_')[1]
    video_id = int(video_name.split('_')[2])

    src_dict = thumos_name_dicts()
    video_id = [src_dict[src_name], video_id]

    # print("video {} person {} camera {} recipe {} : id = {}".format(video_name,person_name,src_name,recipe_name,video_id))

    return video_id


def getVideoName(video_id):
    src_dict = thumos_name_dicts()
    src_dict = {v: k for k, v in src_dict.items()}
    # print('video_id: ', video_id)
    video_name = "video" + "_" + src_dict[video_id[0]] + "_" + str(video_id[1]).zfill(7)

    # print("video {} person {} camera {} recipe {} : id = {}".format(video_id,person_dict[video_id[0]],src_dict[video_id[1]],recipe_dict[video_id[2]],video_name))

    return video_name


# def generate_action_mapping(data_root='/home/jksun/Program/gtad/packages/activitygraph_transformer/data/babel', dataset='babel'):
#

def generate_action_json(data_root='/home/jksun/Program/gtad/packages/activitygraph_transformer/data/babel',
                         dataset='babel',):
    features_root = os.path.join(data_root, 'i3d_feats')
    os.makedirs(features_root, exist_ok=True)
    vid_data_file = os.path.join(data_root, dataset + "_action.json")
    data = store_and_load_babel()

    # Convert raw sequences into snippets
    data['snips'] = np.array(_vids_to_snips(data['X']))
    print('Shape of snippets = ', np.shape(data['snips']))

    # Split data into train, val, test
    spl_idxs = get_split_idxs(data)

    # Select top 60 classes
    all_classes = np.array(data['Psi_c']).reshape(-1)  # (8008, 95) -> (760760,)
    # print('all_classes.shape: ', all_classes.shape)
    counter = Counter(all_classes)
    # print('counter: {}'.format(counter))
    # freq = sorted(Counter(all_classes), key=lambda item: item[1])
    top_60_freq = dict(sorted(counter.items(), key=lambda item: item[1], reverse=True)[2:62]) # [1:61]
    print('freq: ', top_60_freq, 'len(freq): {}'.format(len(top_60_freq)))

    # mapping_file
    mapping_file = os.path.join(data_root, "action_mapping.txt")
    al_to_idx_fpath = '/home/jksun/Program/gtad/data/babel_annotations/action_label_to_idx.json'
    with open(al_to_idx_fpath, 'r') as infile:
        al_to_idx = json.load(infile)
    # print('al_to_idx: ', al_to_idx)
    idx_to_al = {}
    for i, (k, v) in enumerate(al_to_idx.items()):
        idx_to_al[v] = k
    # print('idx_to_al: ', idx_to_al)

    top_60_mapping_str = {}
    top_60_mapping_id = {}
    for i, (k, v) in enumerate(top_60_freq.items()):
        top_60_mapping_id[k] = i
        top_60_mapping_str[i] = idx_to_al[k]
    print('top_60_mapping: ', top_60_mapping_str)

    # raise NotImplementedError
    # Get loaders for each split
    # dsets, dloaders = {}, {}
    # subsets = ['train', 'val', 'test']
    subsets = ['train', 'test']
    action_json = OrderedDict()
    idx_video = 0
    num_queries_list = []  # number of action queries, ie detection slot. This is the maximal number of actions that can be detected in a video.
    for subset in subsets:
        idxs = np.array(spl_idxs[subset])
        spl_X = np.array(data['snips'][idxs])
        spl_Psi_ts = np.array(data['Psi_ts'][idxs])
        spl_Psi_te = np.array(data['Psi_te'][idxs])
        spl_Psi_c = np.array(data['Psi_c'][idxs])
        spl_vids = np.array(data['vids'][idxs])
        for i, (video_name, feat_array, start_array, end_array, action_array) in enumerate(zip(spl_vids, spl_X, spl_Psi_ts, spl_Psi_te, spl_Psi_c)):
            if idx_video % 1000 == 0:
                print('Getting video %d / %d' % (idx_video, len(spl_vids)), flush=True)
            video_name = video_name.replace('https://crichton.is.tue.mpg.de/', '').replace('/', '-')
            if subset in ['val', 'train']:
                subset_name = 'validation'
            else:
                subset_name = subset
            video_name = 'video_{}_{:07d}_{}'.format(subset_name, idx_video, video_name)
            if subset == 'test':
                label_subset = 'testing'
            elif subset in ['train', 'val']:
                label_subset = 'training'

            action_list = []
            T = max(end_array)
            related_fps = T / 100  # Relative frame rate of 100 fps  # each_action['fps']
            # fps = 1  # 30
            # real_fps = int(fps / fixed_fps)  # fps is related to fixed fps = 30
            # assert ((start_array != -1) == (end_array != -1) == (action_array != -1)).all(), '(start_array != -1) == (end_array != -1) == (action_array != -1) should satisfied.'
            for (start_frame, end_frame, act_cat) in zip(start_array, end_array, action_array):
                if act_cat not in top_60_freq:
                    # print('act_cat {} not in top_60_freq'.format(act_cat))
                    continue
                elif start_frame != -1 and end_frame != -1 and act_cat != -1:
                    # print('start_frame: {}, end_frame: {}'.format(start_frame, end_frame))
                    action_list.append(
                        [int(top_60_mapping_id[int(act_cat)]), round(start_frame / related_fps, 1), min(round(feat_array.shape[0] - 1.0, 1), round(end_frame / related_fps, 1))]
                    )
                elif start_frame == -1 and end_frame == -1 and act_cat == -1:
                    pass
                else:
                    pdb.set_trace()
                    print('(start_array != -1) == (end_array != -1) == (action_array != -1) should satisfied.')
            if len(action_list) > 0:
                action_list = action_list[:10]
                num_queries_list.append(len(action_list))
                label = {"duration": round(100-1.0, 2), "subset": label_subset, "actions": action_list}
                # print(label)
                action_json['{}'.format(video_name)] = label
                # print('feat_array.shape: ', feat_array.shape) # (100, 69)
                # padding_feat_array = np.concatenate((feat_array, np.zeros((feat_array.shape[0], 69 - feat_array.shape[1]))), axis=1)
                padding_feat_array = feat_array
                # print('padding_feat_array.shape: ', padding_feat_array.shape)  # (100, 69)
                np.save(os.path.join(features_root, video_name+'.npy'), padding_feat_array[:, None, None, :])  #  (time_step, H, W, feat_dim)
            idx_video = idx_video + 1

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
    with open(vid_data_file, 'w') as f:
        json.dump(action_json, f)

    # if not os.path.exists(mapping_file):
    with open(mapping_file, 'w') as f:
        for k, v in top_60_mapping_str.items():
            f.write("{} {}\n".format(k, v.replace(' ', '_')))


def generate_json_from_AR(data_root='/home/jksun/Program/gtad/packages/activitygraph_transformer/data/babel',
                         dataset='babel',
                         AR_feature_root='/home/jksun/Program/gtad/packages/2s_agcn/work_dir/babel/saved_feature',
                         AR_label_root='/home/jksun/Program/gtad/packages/AR-Shift-GCN/data/babel',
                         num_frame_per_sequence=150,
                         make_stat=False,
                         ):
    features_root = os.path.join(data_root, 'i3d_feats')
    os.makedirs(features_root, exist_ok=True)
    vid_data_file = os.path.join(data_root, dataset + "_action.json")

    # mapping_file
    mapping_file = os.path.join(data_root, "action_mapping.txt")
    al_to_idx_fpath = '/home/jksun/Program/gtad/packages/AR-Shift-GCN/data/babel/action_label_to_idx.json'
    with open(al_to_idx_fpath, 'r') as infile:
        al2idx = json.load(infile)
    idx2al = {al2idx[k]: k for k in al2idx}
    bg_idx = max(al2idx.values())
    idx2al[bg_idx] = 'background'

    top_60_mapping_str = idx2al
    print('top_60_mapping: ', top_60_mapping_str)

    subsets = ['train', 'test']
    action_json = OrderedDict()
    idx_video = 0
    num_queries_list = []  # number of action queries, ie detection slot. This is the maximal number of actions that can be detected in a video.
    for subset in subsets:
        with open(os.path.join(AR_feature_root, '{}_feat.pkl'.format(subset)), 'rb') as f:
            feat_dict = pickle.load(f)
            spl_X = feat_dict['feat']
            gt_label = feat_dict['gt_label']
            pred_label = feat_dict['pred_label']
            index = feat_dict['index']
            print('spl_X.shape: ', spl_X.shape)  # spl_X.shape:  (27838, 256)
        with open('{0}/{1}_label.pkl'.format(AR_label_root, subset), 'rb') as f:
            (spl_vids, spl_Psi_c) = pickle.load(f)
            print('spl_Psi_c.shape: ', spl_Psi_c.shape)  # spl_Psi_c.shape:  (27838,)
            print('spl_vids.shape: ', spl_vids.shape, spl_vids[:10])  # spl_vids.shape:  (27838,)
        with open('{0}/{1}_start_end_time.pkl'.format(AR_label_root, subset), 'rb') as f:
            start_end_dict = pickle.load(f)
            spl_Psi_ts = start_end_dict['start_frame']
            print('spl_Psi_ts.shape: ', spl_Psi_ts.shape)  # spl_Psi_ts.shape:  (27838,)
            spl_Psi_te = start_end_dict['end_frame']
            print('spl_Psi_te.shape: ', spl_Psi_te.shape)  # spl_Psi_te.shape:  (27838,)

        for i, (video_name, feat_array, start_array, end_array, action_array) in enumerate(zip(spl_vids, spl_X, spl_Psi_ts, spl_Psi_te, spl_Psi_c)):
            if idx_video % 1000 == 0:
                print('Getting video %d / %d' % (idx_video, len(spl_vids)), flush=True)
            video_name = video_name.replace('https://crichton.is.tue.mpg.de/', '').replace('/', '-')
            if subset in ['val', 'train']:
                subset_name = 'validation'
            else:
                subset_name = subset
            video_name = 'video_{}_{:07d}_{}'.format(subset_name, idx_video, video_name)
            if subset == 'test':
                label_subset = 'testing'
            elif subset in ['train', 'val']:
                label_subset = 'training'

            action_list = []
            feat_array = feat_array[None, :].repeat(num_frame_per_sequence, axis=0)
            action_list.append(
                [int(action_array), round(start_array * 1., 1), min(round(feat_array.shape[0] - 1.0, 1), round(end_array * 1., 1))]
            )
            if len(action_list) > 0:
                # action_list = action_list[:10]
                num_queries_list.append(len(action_list))
                label = {"duration": round(num_frame_per_sequence-1.0, 2), "subset": label_subset, "actions": action_list}
                # print(label)
                action_json['{}'.format(video_name)] = label
                # print('feat_array.shape: ', feat_array.shape) # (100, 69)
                # padding_feat_array = np.concatenate((feat_array, np.zeros((feat_array.shape[0], 69 - feat_array.shape[1]))), axis=1)
                padding_feat_array = feat_array
                # print('padding_feat_array.shape: ', padding_feat_array.shape)  # (100, 69)
                np.save(os.path.join(features_root, video_name+'.npy'), padding_feat_array[:, None, None, :])  #  (time_step, H, W, feat_dim)
            idx_video = idx_video + 1

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
    with open(vid_data_file, 'w') as f:
        json.dump(action_json, f)

    # if not os.path.exists(mapping_file):
    with open(mapping_file, 'w') as f:
        for k, v in top_60_mapping_str.items():
            f.write("{} {}\n".format(k, v.replace(' ', '_')))

def generate_json_from_SMPL_AR_chunk(data_root='/home/jksun/Program/gtad/packages/activitygraph_transformer/data/babel',
                                dataset='babel',
                                AR_feature_root='/home/jksun/Program/gtad/packages/2s_agcn/work_dir/babel/saved_feature',
                                AR_label_root='/home/jksun/Program/gtad/packages/AR-Shift-GCN/data/babel',
                                num_chunks_per_sequence=100,
                                make_stat=True,
                                ):
    features_root = os.path.join(data_root, 'i3d_feats')
    os.makedirs(features_root, exist_ok=True)
    vid_data_file = os.path.join(data_root, dataset + "_action.json")

    # mapping_file
    mapping_file = os.path.join(data_root, "action_mapping.txt")
    idx2al = {0: 'walk', 1: 'stand', 2: 'turn', 3: 'run', 4: 'step', 5: 'jog', 6: 'throw', 7: 'grasp object',
              8: 'grab person / body part', 9: 'jump', 10: 'place something', 11: 'stretch', 12: 'kick',
              13: 'circular movement', 14: 'take/pick something up', 15: 'bend', 16: 'sit', 17: 'leave',
              18: 'dance', 19: 'interact with / use object'}
    al2idx = {idx2al[k]: k for k in idx2al}

    top_60_mapping_str = idx2al
    print('top_60_mapping: ', top_60_mapping_str)

    with open(os.path.join(AR_feature_root, '{}_feat.pkl'.format('chunk')), 'rb') as f:
        feat_dict = pickle.load(f)
    spl_X = feat_dict['feat']  # feat_array.shape: (808900, 256)
    spl_X = spl_X.reshape(int(spl_X.shape[0] // num_chunks_per_sequence), num_chunks_per_sequence, spl_X.shape[1])
    # gt_label = feat_dict['gt_label']
    # pred_label = feat_dict['pred_label']
    # index = feat_dict['index']
    print('spl_X.shape: ', spl_X.shape)  # spl_X.shape:  (27838, 256)
    with open('{0}/{1}.pkl'.format(AR_label_root, 'babel_seqs_labels'), 'rb') as f:
        data = pickle.load(f)
    Psi = data['Psi']
    spl_Psi_c = []
    spl_Psi_ts = []
    spl_Psi_te = []
    for each_psi in Psi:
        each_psi_c = []
        each_psi_ts = []
        each_psi_te = []
        for item in each_psi:
            # print('item: ', item)
            if item['c'] not in al2idx:
                # print('item[c] {} not in al2idx'.format(item['c']))
                continue
            each_psi_c.append(al2idx[item['c']])
            each_psi_ts.append(item['ts'] * num_chunks_per_sequence)
            each_psi_te.append(item['te'] * num_chunks_per_sequence)
        spl_Psi_c.append(each_psi_c)
        spl_Psi_ts.append(each_psi_ts)
        spl_Psi_te.append(each_psi_te)
    spl_vids = data['vids']
    ann_idx = data['ann_idx']

    print('spl_vids.shape: ', len(spl_vids))  # spl_vids.shape:  (27838,)
    print('ann_idx.shape: ', len(ann_idx))
    spl_idxs = get_split_idxs(data)


    # subsets = ['train', 'test']
    subsets = ['test']
    action_json = OrderedDict()
    idx_video = 0
    num_queries_list = []  # number of action queries, ie detection slot. This is the maximal number of actions that can be detected in a video.
    class_seg_num = {'train': {}, 'val': {}, 'test': {}}
    for subset in subsets:
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
            if subset in ['val', 'train']:
                subset_name = 'validation'
            else:
                subset_name = subset
            video_name = 'video_{}_{:07d}_{}'.format(subset_name, idx_video, video_name)
            if subset == 'test':
                label_subset = 'testing'
            elif subset in ['train', 'val']:
                label_subset = 'training'

            action_list = []
            # feat_array = feat_array[None, :].repeat(num_chunks_per_sequence, axis=0)
            for (start_frame, end_frame, act_cat) in zip(start_array, end_array, action_array):
                # act_cat = 0
                action_list.append(
                    [int(act_cat), round(start_frame * 1., 1), min(round(feat_array.shape[0] - 1.0, 1), round(end_frame * 1., 1))]
                )
                if idx2al[act_cat] not in class_seg_num[subset]:
                    class_seg_num[subset][idx2al[act_cat]] = 1
                else:
                    class_seg_num[subset][idx2al[act_cat]] += 1

            if len(action_list) > 0:
                # action_list = action_list[:10]
                num_queries_list.append(len(action_list))
                label = {"duration": round(num_chunks_per_sequence - 1.0, 2), "subset": label_subset, "actions": action_list}
                # print(label)
                action_json['{}'.format(video_name)] = label
                # print('feat_array.shape: ', feat_array.shape) # (100, 69)
                # padding_feat_array = np.concatenate((feat_array, np.zeros((feat_array.shape[0], 69 - feat_array.shape[1]))), axis=1)
                padding_feat_array = feat_array
                # print('padding_feat_array.shape: ', padding_feat_array.shape)  # (100, 256)
                np.save(os.path.join(features_root, video_name+'.npy'), padding_feat_array[:, None, None, :])  #  (time_step, H, W, feat_dim)
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

    # with open(vid_data_file, 'w') as f:
    #     json.dump(action_json, f)
    #
    # # if not os.path.exists(mapping_file):
    # with open(mapping_file, 'w') as f:
    #     for k, v in top_60_mapping_str.items():
    #         f.write("{} {}\n".format(k, v.replace(' ', '_')))

# def generate_json_from_NTU_AR_chunk(data_root='/home/jksun/Program/gtad/packages/activitygraph_transformer/data/babel',
#                                 dataset='babel',
#                                 # AR_feature_root='/home/jksun/Program/gtad/packages/2s_agcn/work_dir/babel/NTU_chunk_feat/',
#                                 AR_feature_root='/home/jksun/Program/gtad/packages/AR-Shift-GCN/data/babel/babel_ntu_sk_Nv_100_csz_8_ip_to_AR_feat_ext.npy',
#                                 AR_label_root='/home/jksun/Program/gtad/packages/AR-Shift-GCN/data/babel',
#                                 num_chunks_per_sequence=100,
#                                 make_stat=True,
#                                 save_json_and_data=True,
#                                 ):
def generate_json_from_NTU_AR_chunk(
        data_root='../../data/babel',
        dataset='babel',
        # AR_feature_root='/home/jksun/Program/gtad/packages/2s_agcn/work_dir/babel/NTU_chunk_feat/',
        AR_feature_root='../../../AR-Shift-GCN/data/babel/babel_ntu_sk_Nv_100_csz_8_ip_to_AR_feat_ext.npy',
        AR_label_root='../../../AR-Shift-GCN/data/babel',
        num_chunks_per_sequence=100,
        make_stat=True,
        save_json_and_data=True,
        ):
    features_root = os.path.join(data_root, 'i3d_feats')
    os.makedirs(features_root, exist_ok=True)
    vid_data_file = os.path.join(data_root, dataset + "_action.json")

    # mapping_file
    mapping_file = os.path.join(data_root, "action_mapping.txt")
    # idx2als = {0: ['walk'],
    #            1: ['stand'],
    #            2: ['turn'],  # Really short action segments?
    #            3: ['run', 'jog'],
    #            4: ['throw'],
    #            5: ['catch'],
    #            6: ['jump'],
    #            7: ['sit'],
    #            8: ['dance'],  # Large diversity expected
    #            9: ['grasp object', 'place something', 'take/pick something up', 'move something'],
    #            10: ['interact with / use object'],  # Large diversity expected
    #            11: ['lift something'],  # Is this distinct enough from take / pick something up? Yeah, I think so.
    #            12: ['step'],
    #            13: ['stretch', 'yoga'],  # Large diversity expected
    #            14: ['squat'],
    #            15: ['circular movement'],
    #            16: ['bend'],
    #            17: ['kick'],
    #            18: ['hit', 'punch'],
    #            19: ['greet'],
    #            }
    #
    # idx2als_20_59 = {
    #            20: ['t pose'],
    #            21: ['look'],
    #            22: ['grab person / body part'],  # Really short action segments?
    #            23: ['raise body part'],
    #            24: ['leave'],
    #            25: ['a pose'],  # Large diversity expected
    #            26: ['move up/down incline'],
    #            27: ['backwards movement'],  # Large diversity expected
    #            28: ['stumble'],  # Is this distinct enough from take / pick something up? Yeah, I think so.
    #            29: ['hop'],
    #
    #            30: ['perform'],  # Large diversity expected
    #            31: ['lean'],
    #            32: ['arm movement'],
    #            33: ['hand movement'],
    #            34: ['play'],
    #            35: ['shake'],
    #            36: ['kneel'],
    #            37: ['play sport'],  # Large diversity expected
    #            38: ['bow'],  # Is this distinct enough from take / pick something up? Yeah, I think so.
    #            39: ['exercise / training'],
    #
    #            40: ['list body parts'],  # Large diversity expected
    #            41: ['knock'],
    #            42: ['scratch'],
    #            43: ['swing body part'],
    #            44: ['crouch'],
    #            45: ['action with ball'],
    #            46: ['poses'],
    #            47: ['spin'],  # Large diversity expected
    #            48: ['head movements'],  # Is this distinct enough from take / pick something up? Yeah, I think so.
    #            49: ['sports moves'],
    #
    #            50: ['balance'],  # Large diversity expected
    #            51: ['evade'],
    #            52: ['foot movement'],
    #            53: ['lower body part'],
    #            54: ['twist'],
    #            55: ['misc. abstract actions'],
    #            56: ['gesture'],
    #            57: ['rocking movement'],  # Large diversity expected
    #            58: ['communicate (vocalize)'],  # Is this distinct enough from take / pick something up? Yeah, I think so.
    #            59: ['touching face'],
    # }
    #
    # idx2als_60_116 = {
    #            60: ['stop'],  # Large diversity expected
    #            61: ['waist movements'],
    #            62: ['skip'],
    #            63: ['martial arts'],
    #            64: ['move back to original position'],
    #            65: ['give something'],
    #            66: ['face direction'],
    #            67: ['play instrument'],  # Large diversity expected
    #            68: ['tap'],  # Is this distinct enough from take / pick something up? Yeah, I think so.
    #            69: ['touching body parts'],
    #
    #            70: ['cartwheel'],  # Large diversity expected
    #            71: ['make'],
    #            72: ['lie'],
    #            73: ['animal behavior'],
    #            74: ['stagger'],
    #            75: ['point'],
    #            76: ['adjust'],
    #            77: ['drink'],  # Large diversity expected
    #            78: ['crawl'],  # Is this distinct enough from take / pick something up? Yeah, I think so.
    #            79: ['sideways movement'],
    #
    #            80: ['telephone call'],  # Large diversity expected
    #            81: ['clean something'],
    #            82: ['shuffle'],
    #            83: ['dribble'],
    #            84: ['golf'],
    #            85: ['spread'],
    #            86: ['wait'],
    #            87: ['trip'],  # Large diversity expected
    #            88: ['touch object'],  # Is this distinct enough from take / pick something up? Yeah, I think so.
    #            89: ['mime'],
    #
    #            90: ['open something'],  # Large diversity expected
    #            91: ['salute'],
    #            92: ['limp'],
    #            93: ['relax'],
    #            94: ['flap'],
    #            95: ['legs movement'],
    #            96: ['move misc. body part'],
    #            97: ['sudden movement'],  # Large diversity expected
    #            98: ['fight'],  # Is this distinct enough from take / pick something up? Yeah, I think so.
    #            99: ['jumping jacks'],
    #
    #            100: ['clap'],  # Large diversity expected
    #            101: ['duck'],
    #            102: ['rub'],
    #            103: ['rolling motion'],
    #            104: ['sneak'],
    #            105: ['stances'],
    #            106: ['prepare'],
    #            107: ['lunge'],  # Large diversity expected
    #            108: ['fall'],  # Is this distinct enough from take / pick something up? Yeah, I think so.
    #            109: ['misc. activities'],
    #
    #            110: ['rolls on ground'],  # Large diversity expected
    #            111: ['shrug'],
    #            112: ['leap'],
    #            113: ['search'],
    #            }
    #
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
    # idx2als.update(idx2als_20_59)
    # idx2als.update(idx2als_60_116)
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

    top_60_mapping_str = idx2al
    print('top_60_mapping: ', top_60_mapping_str)
    if AR_feature_root.endswith('.npy'):
        spl_X = np.load(AR_feature_root)  # spl_X.shape:  (808900, 3, 8, 25, 1)
        spl_X = spl_X.transpose(0, 2, 1, 3, 4)
        spl_X = spl_X.reshape(spl_X.shape[0], -1)
        spl_X = spl_X.reshape(int(spl_X.shape[0] // num_chunks_per_sequence), num_chunks_per_sequence, spl_X.shape[1])  # spl_X.shape:  (8089, 100, 600)
        print('spl_X.shape: ', spl_X.shape)
    else:
        with open(os.path.join(AR_feature_root, '{}_feat.pkl'.format('epoch1_test')), 'rb') as f:
            feat_dict = pickle.load(f)
        spl_X = feat_dict['feat']  # feat_array.shape: (808900, 256)
        spl_X = spl_X.reshape(int(spl_X.shape[0] // num_chunks_per_sequence), num_chunks_per_sequence, spl_X.shape[1])
    # gt_label = feat_dict['gt_label']
    # pred_label = feat_dict['pred_label']
    # index = feat_dict['index']
    print('spl_X.shape: ', spl_X.shape)  # spl_X.shape:  (27838, 256)
    with open('{0}/{1}.pkl'.format(AR_label_root, 'babel_seqs_labels'), 'rb') as f:
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
            # if item['c'] in ['run', 'jog']:
            #     action_class = 'run or jog'
            # elif item['c'] in ['grasp object', 'place something', 'take/pick something up', 'move something']:
            #     action_class = 'grasp object or place something or take/pick something up or move something'
            # elif item['c'] in ['stretch', 'yoga']:
            #     action_class = 'stretch or yoga'
            # elif item['c'] in ['hit', 'punch']:
            #     action_class = 'hit or punch'
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
                # print('item[c] {} not in al2idx'.format(item['c']))
                continue
            each_psi_c.append(al2idx[action_class])
            each_psi_ts.append(item['ts'] * num_chunks_per_sequence)
            each_psi_te.append(item['te'] * num_chunks_per_sequence)
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

    num_total_videos = 0
    num_ignored_videos = 0
    num_npy_videos = 0
    subsets = ['train', 'val']
    # subsets = ['train', 'test']
    # subsets = ['test']
    action_json = OrderedDict()
    idx_video = 0
    num_queries_list = []  # number of action queries, ie detection slot. This is the maximal number of actions that can be detected in a video.
    class_seg_num = {'train': {}, 'val': {}, 'test': {}}
    for subset in subsets:
        idxs = np.array(spl_idxs[subset])
        for i, (video_name, feat_array, start_array, end_array, action_array, ann_idx_array) \
                in enumerate(zip(spl_vids, spl_X, spl_Psi_ts, spl_Psi_te, spl_Psi_c, ann_idx)):
            num_total_videos += 1
            if ann_idx_array not in idxs:
                num_ignored_videos += 1
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
                    [int(act_cat), round(start_frame * 1., 1), min(round(feat_array.shape[0] - 1.0, 1), round(end_frame * 1., 1))]
                )
                action_list = action_list[:20]
                if idx2al[act_cat] not in class_seg_num[subset]:
                    class_seg_num[subset][idx2al[act_cat]] = 1
                else:
                    class_seg_num[subset][idx2al[act_cat]] += 1

            if len(action_list) > 0:
                # action_list = action_list[:10]
                num_queries_list.append(len(action_list))
                label = {"duration": round(num_chunks_per_sequence - 1.0, 2), "subset": label_subset, "actions": action_list}
                # print(label)
                action_json['{}'.format(video_name)] = label
                # print('feat_array.shape: ', feat_array.shape) # (100, 69)
                # padding_feat_array = np.concatenate((feat_array, np.zeros((feat_array.shape[0], 69 - feat_array.shape[1]))), axis=1)
                padding_feat_array = feat_array
                # print('padding_feat_array.shape: ', padding_feat_array.shape)  # (100, 256)
                if save_json_and_data:
                    np.save(os.path.join(features_root, video_name+'.npy'), padding_feat_array[:, None, None, :])  #  (time_step, H, W, feat_dim)

                num_npy_videos += 1
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
        # if not os.path.exists(mapping_file):
        with open(mapping_file, 'w') as f:
            for k, v in top_60_mapping_str.items():
                f.write("{} {}\n".format(k, v.replace(' ', '_')))
        print('Mapping file saved to {}'.format(mapping_file))
    print('Num ignored videos: {} / {}, '.format(num_ignored_videos, num_total_videos),
          'Num saved npy videos: {} / {}'.format(num_npy_videos, num_total_videos))

def generate_json_from_SMPL(
        data_root='../../data/babel',
        dataset='babel',
        # AR_feature_root='/home/jksun/Program/gtad/packages/2s_agcn/work_dir/babel/NTU_chunk_feat/',
        AR_feature_root='../../../data/babel/babel_smpl_sk_Nv_100_csz_8_joint_groundcontact.npy',
        # AR_label_root='../../../AR-Shift-GCN/data/babel',
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
    # idx2als.update(idx2als_20_59)
    # idx2als.update(idx2als_60_116)
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

    top_60_mapping_str = idx2al
    print('top_60_mapping: ', top_60_mapping_str)
    if AR_feature_root.endswith('.npy'):
        spl_X = np.load(AR_feature_root)  # spl_X.shape:  (808900, 3, 8, 25, 1)
        print('spl_X.shape: ', spl_X.shape)  # spl_X.shape:  (8808, 100, 8, 96)
        # spl_X = spl_X.transpose(0, 2, 1, 3, 4)
        # spl_X = spl_X.reshape(spl_X.shape[0], -1)
        # spl_X = spl_X.reshape(int(spl_X.shape[0] // num_chunks_per_sequence), num_chunks_per_sequence, spl_X.shape[1])  # spl_X.shape:  (8089, 100, 600)
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
    # gt_label = feat_dict['gt_label']
    # pred_label = feat_dict['pred_label']
    # index = feat_dict['index']
    print('spl_X.shape: ', spl_X.shape)  # spl_X.shape:  (27838, 256)
    # with open('{0}/{1}.pkl'.format(AR_label_root, 'babel_seqs_labels'), 'rb') as f:
    #     data = pickle.load(f)

    train_json = os.path.join(babel_label_path, 'train.json')  # train_extra.json
    val_json = os.path.join(babel_label_path, 'val.json')  # val_extra.json
    splits = {'train': [], 'val': []}
    sample_paths = []
    d = {'X': [], 'Psi': [], 'fps': [], 'ft_type': 'pos', 'url': [], 'feat_path': [], 'vids': [], 'amass': []}

    with open(train_json, 'r') as f:
        train_label = json.load(f)
    for i, (k, v) in enumerate(train_label.items()):
        # print('v.keys(): ', v.keys())  # v.keys():  dict_keys(['babel_sid', 'url', 'feat_p', 'dur', 'seq_ann', 'frame_ann'])
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
                    #                 Psi.append({'ts': int(label['start_t'] * 100), 'te': int(label['end_t'] * 100), 'c': each_act_cat})
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
        # if args.mode == 'debug' and i >= first_n_samples:
        #     break

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
        # if args.mode == 'debug' and i >= first_n_samples:
        #     break

    d['ann_idx'] = list(range(len(d['Psi'])))
    data = d

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
            # if item['c'] in ['run', 'jog']:
            #     action_class = 'run or jog'
            # elif item['c'] in ['grasp object', 'place something', 'take/pick something up', 'move something']:
            #     action_class = 'grasp object or place something or take/pick something up or move something'
            # elif item['c'] in ['stretch', 'yoga']:
            #     action_class = 'stretch or yoga'
            # elif item['c'] in ['hit', 'punch']:
            #     action_class = 'hit or punch'
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
                # print('item[c] {} not in al2idx'.format(item['c']))
                continue
            each_psi_c.append(al2idx[action_class])
            each_psi_ts.append(item['ts'] * num_chunks_per_sequence)
            each_psi_te.append(item['te'] * num_chunks_per_sequence)
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
    # spl_idxs = get_split_idxs(data)
    spl_lidxs = defaultdict(list)
    for idx, vid in enumerate(data['vids']):
        vid_spl = [spl for spl in splits if vid in splits[spl]][0]
        # print('vid_spl, idx: ', vid_spl, idx)
        spl_lidxs[vid_spl].append(idx)
    spl_aidxs = {spl: np.array(spl_lidxs[spl]) for spl in spl_lidxs}

    for spl in spl_aidxs:
        print('BABEL dataset # {0} = {1}'.format(spl, len(spl_aidxs[spl])))
    spl_idxs = spl_aidxs

    num_total_videos = 0
    num_ignored_videos = 0
    num_npy_videos = 0
    subsets = ['train', 'val']
    # subsets = ['train', 'test']
    # subsets = ['test']
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
                    [int(act_cat), round(start_frame * 1., 1), min(round(feat_array.shape[0] - 1.0, 1), round(end_frame * 1., 1))]
                )
                action_list = action_list[:20]
                if idx2al[act_cat] not in class_seg_num[subset]:
                    class_seg_num[subset][idx2al[act_cat]] = 1
                else:
                    class_seg_num[subset][idx2al[act_cat]] += 1

            if len(action_list) > 0:
                # action_list = action_list[:10]
                num_queries_list.append(len(action_list))
                label = {"duration": round(num_chunks_per_sequence - 1.0, 2), "subset": label_subset, "actions": action_list}
                # print(label)
                action_json['{}'.format(video_name)] = label
                # print('feat_array.shape: ', feat_array.shape) # (100, 69)
                if feat_array.shape[-1] < feat_array_last_shape:
                    padding_feat_array = np.concatenate((feat_array, np.zeros((feat_array.shape[0], feat_array_last_shape - feat_array.shape[1]))), axis=1)
                else:
                    padding_feat_array = feat_array
                # print('padding_feat_array.shape: ', padding_feat_array.shape)  # (100, 256)
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
        # if not os.path.exists(mapping_file):
        with open(mapping_file, 'w') as f:
            for k, v in top_60_mapping_str.items():
                f.write("{} {}\n".format(k, v.replace(' ', '_')))
        print('Mapping file saved to {}'.format(mapping_file))
    print('Num ignored videos: {} / {}, '.format(num_ignored_videos, num_total_videos),
          'Num saved npy videos: {} / {}'.format(num_npy_videos, num_total_videos))

def generate_json_from_hongwei(
        data_root='data/babel',
        dataset='babel',
        # AR_feature_root='/home/jksun/Program/gtad/packages/2s_agcn/work_dir/babel/NTU_chunk_feat/',
        # AR_feature_root='../../../data/babel/babel_smpl_sk_Nv_100_csz_8_joint_groundcontact.npy',
        AR_feature_root='data/babel/babel_smpl_sk_Nv_100_csz_8_ip_to_AR_feat_ext_vhongwei.npy',
        # AR_label_root='../../../AR-Shift-GCN/data/babel',
        babel_label_path='data/hongwei',
        npz_prefix='data/hongwei',
        amass_unzip_prefix='data/hongwei',
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
    # idx2als.update(idx2als_20_59)
    # idx2als.update(idx2als_60_116)
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

    top_60_mapping_str = idx2al
    print('top_60_mapping: ', top_60_mapping_str)
    if AR_feature_root.endswith('.npy'):
        spl_X = np.load(AR_feature_root)  # spl_X.shape:  (808900, 3, 8, 25, 1)
        print('spl_X.shape: ', spl_X.shape)  # spl_X.shape:  (8808, 100, 8, 96)
        # spl_X = spl_X.transpose(0, 2, 1, 3, 4)
        # spl_X = spl_X.reshape(spl_X.shape[0], -1)
        # spl_X = spl_X.reshape(int(spl_X.shape[0] // num_chunks_per_sequence), num_chunks_per_sequence, spl_X.shape[1])  # spl_X.shape:  (8089, 100, 600)
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
    # gt_label = feat_dict['gt_label']
    # pred_label = feat_dict['pred_label']
    # index = feat_dict['index']
    print('spl_X.shape: ', spl_X.shape)  # spl_X.shape:  (27838, 256)
    # with open('{0}/{1}.pkl'.format(AR_label_root, 'babel_seqs_labels'), 'rb') as f:
    #     data = pickle.load(f)

    train_json = os.path.join(babel_label_path, 'train.json')  # train_extra.json
    val_json = os.path.join(babel_label_path, 'val.json')  # val_extra.json
    splits = {'train': [], 'val': []}
    sample_paths = []
    d = {'X': [], 'Psi': [], 'fps': [], 'ft_type': 'pos', 'url': [], 'feat_path': [], 'vids': [], 'amass': []}

    with open(train_json, 'r') as f:
        train_label = json.load(f)
    for i, (k, v) in enumerate(train_label.items()):
        amass_feats = v['feat_p']
        # npz_file = np.load(os.path.join(npz_prefix, amass_feats.replace('.npz', '.npy')))
        npz_file_name_list = sorted(os.listdir(os.path.join(npz_prefix, amass_feats)))
        npz_file_list = []
        for each_npz_file_name in npz_file_name_list:
            each_npz_file = np.load(os.path.join(npz_prefix, amass_feats, each_npz_file_name))
            each_npz_file = each_npz_file[:24].reshape(1, 24, 3)
            # print('each_npz_file.shape: ', each_npz_file.shape)  # (45, 3)
            npz_file_list.append(each_npz_file)
        # npz_file = np.load(os.path.join(npz_prefix, amass_feats.replace('.npz', '.npy')))
        npz_file = np.array(npz_file_list)
        amass_npz = {'mocap_framerate': 30}  # np.load(os.path.join(amass_unzip_prefix, amass_feats))

        d['X'].append(copy.deepcopy(npz_file.reshape(npz_file.shape[0], -1)))
        sample_paths.append(os.path.join(amass_unzip_prefix, amass_feats))
        try:
            labels = v['frame_ann']['labels']
            Psi = []
            for label in labels:
                for each_act_cat in label['act_cat']:
                    #                 Psi.append({'ts': int(label['start_t'] * 100), 'te': int(label['end_t'] * 100), 'c': each_act_cat})
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
            # print([k for k in amass_npz.files], amass_npz['mocap_framerate'])
        # if args.mode == 'debug' and i >= first_n_samples:
        #     break

    print('Val Set')
    with open(val_json, 'r') as f:
        val_label = json.load(f)
    for i, (k, v) in enumerate(val_label.items()):
        amass_feats = v['feat_p']
        # npz_file = np.load(os.path.join(npz_prefix, amass_feats.replace('.npz', '.npy')))
        npz_file_name_list = sorted(os.listdir(os.path.join(npz_prefix, amass_feats)))
        npz_file_list = []
        for each_npz_file_name in npz_file_name_list:
            each_npz_file = np.load(os.path.join(npz_prefix, amass_feats, each_npz_file_name))
            each_npz_file = each_npz_file[:24].reshape(1, 24, 3)
            # print('each_npz_file.shape: ', each_npz_file.shape)  # (45, 3)
            npz_file_list.append(each_npz_file)
        # npz_file = np.load(os.path.join(npz_prefix, amass_feats.replace('.npz', '.npy')))
        npz_file = np.array(npz_file_list)
        amass_npz = {'mocap_framerate': 30}  # np.load(os.path.join(amass_unzip_prefix, amass_feats))

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
            # print([k for k in amass_npz.files], amass_npz['mocap_framerate'])
        # if args.mode == 'debug' and i >= first_n_samples:
        #     break

    d['ann_idx'] = list(range(len(d['Psi'])))
    data = d

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
            # if item['c'] in ['run', 'jog']:
            #     action_class = 'run or jog'
            # elif item['c'] in ['grasp object', 'place something', 'take/pick something up', 'move something']:
            #     action_class = 'grasp object or place something or take/pick something up or move something'
            # elif item['c'] in ['stretch', 'yoga']:
            #     action_class = 'stretch or yoga'
            # elif item['c'] in ['hit', 'punch']:
            #     action_class = 'hit or punch'
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
                # print('item[c] {} not in al2idx'.format(item['c']))
                continue
            each_psi_c.append(al2idx[action_class])
            each_psi_ts.append(item['ts'] * num_chunks_per_sequence)
            each_psi_te.append(item['te'] * num_chunks_per_sequence)
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
    # spl_idxs = get_split_idxs(data)
    spl_lidxs = defaultdict(list)
    for idx, vid in enumerate(data['vids']):
        vid_spl = [spl for spl in splits if vid in splits[spl]][0]
        # print('vid_spl, idx: ', vid_spl, idx)
        spl_lidxs[vid_spl].append(idx)
    spl_aidxs = {spl: np.array(spl_lidxs[spl]) for spl in spl_lidxs}

    for spl in spl_aidxs:
        print('BABEL dataset # {0} = {1}'.format(spl, len(spl_aidxs[spl])))
    spl_idxs = spl_aidxs

    num_total_videos = 0
    num_ignored_videos = 0
    num_npy_videos = 0
    subsets = ['train', 'val']
    # subsets = ['train', 'test']
    # subsets = ['test']
    action_json = OrderedDict()
    idx_video = 0
    num_queries_list = []  # number of action queries, ie detection slot. This is the maximal number of actions that can be detected in a video.
    class_seg_num = {'train': {}, 'val': {}, 'test': {}}
    for subset in subsets:
        try:
            idxs = np.array(spl_idxs[subset])
        except:
            print('{} does not exist'.format(subset))
            continue
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
                    [int(act_cat), round(start_frame * 1., 1), min(round(feat_array.shape[0] - 1.0, 1), round(end_frame * 1., 1))]
                )
                action_list = action_list[:20]
                if idx2al[act_cat] not in class_seg_num[subset]:
                    class_seg_num[subset][idx2al[act_cat]] = 1
                else:
                    class_seg_num[subset][idx2al[act_cat]] += 1

            if len(action_list) > 0:
                # action_list = action_list[:10]
                num_queries_list.append(len(action_list))
                label = {"duration": round(num_chunks_per_sequence - 1.0, 2), "subset": label_subset, "actions": action_list}
                # print(label)
                action_json['{}'.format(video_name)] = label
                # print('feat_array.shape: ', feat_array.shape) # (100, 69)
                if feat_array.shape[-1] < feat_array_last_shape:
                    padding_feat_array = np.concatenate((feat_array, np.zeros((feat_array.shape[0], feat_array_last_shape - feat_array.shape[1]))), axis=1)
                else:
                    padding_feat_array = feat_array
                # print('padding_feat_array.shape: ', padding_feat_array.shape)  # (100, 256)
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
        # if not os.path.exists(mapping_file):
        with open(mapping_file, 'w') as f:
            for k, v in top_60_mapping_str.items():
                f.write("{} {}\n".format(k, v.replace(' ', '_')))
        print('Mapping file saved to {}'.format(mapping_file))
    print('Num ignored videos: {} / {}, '.format(num_ignored_videos, num_total_videos),
          'Num saved npy videos: {} / {}'.format(num_npy_videos, num_total_videos))

if __name__ == "__main__":
    # generate_action_mapping()
    # generate_action_json()
    # generate_json_from_AR()
    # generate_json_from_SMPL_AR_chunk()
    # generate_json_from_NTU_AR_chunk()
    # generate_json_from_SMPL()  # Default
    generate_json_from_hongwei()
