import sys, os, pdb
import os.path as osp
import numpy as np
import json, pickle, csv


def read_json(json_filename):
	'''Return contents of JSON file'''
	jc = None
	with open(json_filename) as infile:
		jc = json.load(infile)
	return jc


def write_json(contents, json_filename):
	'''Write contents into JSON file'''
	with open(json_filename, 'w') as outfile:
		json.dump(contents, outfile)
	return None


def read_pkl(pkl_filename):
    '''Return contents of pikcle file'''
    pklc = None
    with open(pkl_filename, 'rb') as infile:
        pklc = pickle.load(infile)
    return pklc


def smpl_to_nturgbd(model_type='smplh', out_format='nturgbd'):
    ''' Borrowed from https://gitlab.tuebingen.mpg.de/apunnakkal/2s_agcn/-/blob/master/data_gen/smpl_data_utils.py
    NTU mapping
    -----------
    0 --> ?
    1-base of the spine
    2-middle of the spine
    3-neck
    4-head
    5-left shoulder
    6-left elbow
    7-left wrist
    8-left hand
    9-right shoulder
    10-right elbow
    11-right wrist
    12-right hand
    13-left hip
    14-left knee
    15-left ankle
    16-left foot
    17-right hip
    18-right knee
    19-right ankle
    20-right foot
    21-spine
    22-tip of the left hand
    23-left thumb
    24-tip of the right hand
    25-right thumb

    :param model_type:
    :param out_format:
    :return:
    '''
    if model_type == 'smplh' and out_format == 'nturgbd':
        '22 and 37 are approximation for hand (base of index finger)'
        return np.array([0, 3, 12, 15,
                         16, 18, 20, 22,        #left hand
                         17, 19, 21, 37,           # right hand
                         1, 4, 7, 10,           #left leg
                         2, 5, 8, 11,           #right hand
                         9,
                         63, 64 , 68, 69
                         ],
                        dtype=np.int32)

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
    al_to_idx_fpath = '/ps/project/conditional_action_gen/AR-Shift-GCN/data/babel/action_label_to_idx.json'
    al_to_idx = {}
    if osp.exists(al_to_idx_fpath):
        with open(al_to_idx_fpath, 'r') as infile:
            al_to_idx = json.load(infile)
    else:
        set_als = set([psi['c'] for Psi in data['Psi'] for psi in Psi])
        al_to_idx = {al: alidx for alidx, al in enumerate(set_als)}
        with open(al_to_idx_fpath, 'w') as outfile:
            json.dump(al_to_idx, outfile)

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