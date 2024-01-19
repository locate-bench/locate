#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 achandrasekaran <arjun.chandrasekaran@tuebingen.mpg.de>
#
# Distributed under terms of the MIT license.

import sys, os, pdb
from os.path import join as ospj
import json, pickle
import argparse
import re

import numpy as np

sys.path.append('../../../tools/')
import dutils

"""
Code to visualize (final) results after training
"""


def get_idx2al():
    '''Return category idx --> action category string mapping'''
    fp = '../../../AR-Shift-GCN/data/babel/action_label_to_idx.json'
    al2idx = dutils.read_json(fp)
    idx2al = {v: k for k, v in al2idx.items()}
    return idx2al


def get_agtidx2url_map():
    ''' '''
    # URL prefix
    url_pre = 'https://crichton.is.tue.mpg.de/hmotionlabeling'
    # Read AGT data file
    jc = dutils.read_json('../../data/babel/babel_action.json')
    re_str = 'video_test_(\d+)_hmotionlabeling-(\d+).mp4@(\d+)'

    # Get mapping for each test video in AGT
    vstr2url, uniquestr2url = {}, {}
    n_dup_ann = 0  # Count # duplicate anns in test set
    for v in jc:
        if 'video_test' not in v:
            continue
        # AGT vid idx, BABEL url idx, AGT unique annotation idx
        agt_i, url_i, ann_i = re.match(re_str, v).groups()
        url = '{0}/{1}.mp4'.format(url_pre, url_i)
        # Map prediction files keys to URLs. NOTE: Duplicates are overwritten.
        vstr2url[f'video_test_{agt_i}'] = url
        uniquestr2url[v] = url
    print('# video_strs / # unique seqs. = {0}/{1}'.format(len(vstr2url), len(uniquestr2url)))

    return vstr2url, uniquestr2url


def load_preds(fp, idx2al, vstr2url):
    '''Load predictions from the model, process it.
    '''
    max_dur = 99.0
    pred_jc = dutils.read_json(fp)
    vstrs = pred_jc['results'].keys()
    print('Loaded {0} predictions from {1}'.format(len(vstrs), fp))
    preds = {}
    for vstr in vstrs:
        url = vstr2url[vstr]
        Psi_hat = []
        for psi in pred_jc['results'][vstr]:
            # Time in range [0, 1]
            Psi_hat.append({'action': idx2al[psi['label']],
                            'start': max(psi['segment'][0] / max_dur, 0.0),
                            'end': min(psi['segment'][1] / max_dur, 1.0)
                            })
        preds[url] = Psi_hat
    return preds


def load_gt(fp, idx2al, ustr2url):
    '''Load test GT data, process it.
    Return:
        idx2al (dict): {act cat idx <int>: act cat <str>, ... }
        gtv (dict): {'vid': [
                      {'action': 'walk', 'start': <float>, 'end': <float>},
                      {'action': 'stand', ...}, ...],
                    ...}
    '''
    max_dur = 99.0
    print('Loading GT from ', fp)
    gt = dutils.read_json(fp)
    # Convert to desired format
    gtv = {}
    for ustr in gt:
        if 'video_test' not in ustr:
            continue
        Psi = []
        for tup in gt[ustr]['actions']:
            # NOTE: Start, end are normalized to range [0, 1]
            Psi.append({'action': idx2al[tup[0]],
                        'start': max(tup[1] / max_dur, 0.0),
                        'end': min(tup[2] / max_dur, 1.0)
                        })
        url = ustr2url[ustr]
        # TODO: Handle this case -- viz. max. mAP seq.
        if url in gtv:
            # print('Ignoring repeated annotation in GT: ', url)
            pass
        gtv[url] = Psi

    return gtv


def collate_gt_preds(gt, preds, n_seqs):
    ''' '''
    # Choose n_seqs random seqs to visualize
    l_urls = list(preds.keys())
    np.random.seed(123)
    rand_i = np.random.randint(low=0, high=len(l_urls), size=n_seqs)
    # Loop over each seq.
    viz_ds = {}
    for i in rand_i:
        url = l_urls[i]
        viz_ds[url] = {'epoch': -1,
                       'gt_segments': gt[url],
                       'pred_segments': preds[url]}
    return viz_ds


def dump_new_viz_page(viz_ds, output_dir):
    ''' '''
    # Integrate this DS with viz. webpage
    webp = open('../../../3D-TAL/viz/viz_webpage_template.html').read()
    new_webp = webp.replace(r'var vid_act_map = {}', 'var vid_act_map = {0}'.format(viz_ds))
    with open(ospj(output_dir, 'preds_viz_tal.html'), 'w') as outf:
        outf.write(new_webp)
    print('Save to {}'.format(ospj(output_dir, 'preds_viz_tal.html')))
    return None


def main():
    # Parse all command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        help='Path to folder where model preds are stored.',
                        # default='checkpoints_numqueries20_lr1e-5_lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads8_nenclayers4_ndeclayers4_hdim256_sr1_batchsize64_nposembdict512_numinputs100_conc_hl-level_feats')
                        default='checkpoints_numqueries100_lr1e-5_lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers4_hdim256_sr1_batchsize1_nposembdict512_numinputs100')
    parser.add_argument('--ckpt_epoch', type=str, default='95',
                        help='Epoch of ckpt whose results we want to load.')
    parser.add_argument('--n_viz', type=int, default=30,
                        help='# seqs. whose predictions are visualized.')
    args = parser.parse_args()

    # Determine paths to GT, Predictions from args
    cur_p = os.getcwd()
    rel_ckpt_p = '../../output/checkpoints_/'
    ckpt_fold_p = ospj(cur_p, rel_ckpt_p, args.output_dir)
    # gt_labels_p = ospj(ckpt_fold_p, f'{args.ckpt_epoch}_babel_groundtruth_result_agt.pkl')
    gt_labels_p = '../../data/babel/babel_action.json'
    pred_labels_p = ospj(ckpt_fold_p, f'{args.ckpt_epoch}_babel_detection_result_agt.json')

    # Load video_test_<> --> URL map
    vstr2url, ustr2url = get_agtidx2url_map()

    # Load GT, Predictions
    idx2al = get_idx2al()
    preds = load_preds(pred_labels_p, idx2al, vstr2url)
    gt = load_gt(gt_labels_p, idx2al, ustr2url)

    # Collate preds, gt
    viz_ds = collate_gt_preds(gt, preds, args.n_viz)

    # Write viz. webpage
    dump_new_viz_page(viz_ds, ckpt_fold_p)


if __name__ == '__main__':
    main()
