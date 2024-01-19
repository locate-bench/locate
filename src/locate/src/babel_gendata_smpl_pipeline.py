import torch
import sys, os, pdb
import numpy as np
import json
import copy

# Custom
import sys

sys.path.extend(['../babel_tools/'])
from rotation import *

import preprocess
import viz
deploy_to = 'BJ_IDC'
if deploy_to == 'devcloud':
    path_prefix = '/data'
elif deploy_to == 'BJ_IDC':
    path_prefix = 'data/hongwei'
else:
    path_prefix = os.path.expanduser("~")
# 1_2_1104_Female2_0_smpl_numpy  armchair001_stageII_smpl_numpy
babel_label_path = path_prefix
train_json = os.path.join(babel_label_path, 'train.json')  # train_extra.json
val_json = os.path.join(babel_label_path, 'val.json')  # val_extra.json
npz_prefix = path_prefix
amass_unzip_prefix = path_prefix

print('Training Set')

d = {'X': [], 'Psi': [], 'fps': [], 'ft_type': 'pos', 'url': [], 'feat_path': [], 'vids': []}
splits = {'train': [], 'val': []}

with open(train_json, 'r') as f:
    train_label = json.load(f)
for i, (k,v) in enumerate(train_label.items()):
    amass_feats = v['feat_p']
    npz_file_name_list = sorted(os.listdir(os.path.join(npz_prefix, amass_feats)))
    npz_file_list = []
    for each_npz_file_name in npz_file_name_list:
        each_npz_file = np.load(os.path.join(npz_prefix, amass_feats, each_npz_file_name))
        # print('each_npz_file.shape: ', each_npz_file.shape)
        each_npz_file = each_npz_file[:24].reshape(1, 24, 3)
        npz_file_list.append(each_npz_file)
    # npz_file = np.load(os.path.join(npz_prefix, amass_feats.replace('.npz', '.npy')))
    npz_file = np.array(npz_file_list)
    print('npz_file.shape: ', npz_file.shape)
    amass_npz = {'mocap_framerate': 30}   # np.load(os.path.join(amass_unzip_prefix, amass_feats))
    d['X'].append(copy.deepcopy(npz_file.reshape(npz_file.shape[0], -1)))
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
        # print([k for k in amass_npz.files], amass_npz['mocap_framerate'])

print('Val Set')
with open(val_json, 'r') as f:
    val_label = json.load(f)
for i, (k,v) in enumerate(val_label.items()):
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
    print('npz_file.shape: ', npz_file.shape)
    amass_npz = {'mocap_framerate': 30}  # np.load(os.path.join(amass_unzip_prefix, amass_feats))
    d['X'].append(copy.deepcopy(npz_file.reshape(npz_file.shape[0], -1)))
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
d['ann_idx'] = list(range(len(d['Psi'])))

print('d.keys(): ', d.keys())
print('Type of feats stored in object: ', d['ft_type'])
print('Total # untrimmed sequences = ', len(d['X']))
len_els = {k: len(d[k]) for k in d.keys()}
print('len_els: ', len_els)
# Print 1 seq. data
print('d[X][0].shape: ', d['X'][0].shape)
print('d[Psi][0]: ', d['Psi'][0])
print('d[fps][0]: ', d['fps'][0])
d_ntu = d
len_els = {k: len(d_ntu[k]) for k in d_ntu.keys()}

def prep_x_for_norm(x, is_trans):
    print('1 x.shape: ', x.shape)  # N, T, C * {V+1, V}
    # Ignore translation if included
    if is_trans:
        x = x[:, :, 3:]
        print('2 x.shape: ', x.shape)
    x = x.reshape(x.shape[0], x.shape[1], -1, 3)  # N, T, V, C
    print('3 x.shape: ', x.shape)
    x = x.transpose(0, 3, 1, 2)  # N, C, T, V
    print('4 x.shape: ', x.shape)
    x = x[:, :, :, :, np.newaxis]  # N, C, T, V, M
    print('5 x.shape: ', x.shape)
    return x


def get_chunks(X, fps_info, chunk_sz=8, Nv=100, type='no_scale', specified_fps=30):
    '''Given input
    Arguments:
        X (list): BABEL seqs. each of sz (T, raw_ft_dim)
        chunk_sz (int): Size of window for which features
            are extracted.
        Nv (int): # feats. input to the localization model.
    '''
    # Raw data shape
    n_seqs = len(X)
    _, raw_ft_dim = X[0].shape
    fps_set = set()
    if type == 'no_scale':
        Xch = []
        ch_st_idxs = {}
        clip_idx = 0
        # Chunk each seq.
        for seq_i_idx in range(n_seqs):
            seq_i = X[seq_i_idx]
            fps = fps_info[seq_i_idx]
            # print('seq_i.shape[0]: {}, fps: {}'.format(seq_i.shape[0], fps))
            st_idxs = np.linspace(0, seq_i.shape[0] - int(fps / 30), int(seq_i.shape[0] / fps * 30), dtype=int)
            # print('st_idxs: {}'.format(st_idxs))
            fps_set.add(int(fps))
            # fps_30_rate = int(fps / specified_fps)
            # seq_i = seq_i[::fps_30_rate]
            seq_i = seq_i[st_idxs]
            T, raw_ft_dim = seq_i.shape
            seq_clip_idx = 0
            for step_i in range(0, T, Nv*chunk_sz):
                # print('seq_i[step_i:step_i+Nv*chunk_sz].shape[0]: {}'.format(seq_i[step_i:step_i+Nv*chunk_sz].shape[0]))
                selected_clip = np.concatenate((seq_i[step_i:step_i+Nv*chunk_sz],
                                                np.zeros((Nv*chunk_sz - seq_i[step_i:step_i+Nv*chunk_sz].shape[0], raw_ft_dim))), axis=0)
                # selected_T = selected_clip.shape[0]
                # if selected_T < Nv * chunk_sz:
                selected_clip = np.reshape(selected_clip, (Nv, chunk_sz, raw_ft_dim))
                Xch.append(selected_clip)
                ch_st_idxs[clip_idx] = [0, seq_i[step_i:step_i+Nv*chunk_sz].shape[0], T, seq_clip_idx]
                clip_idx = clip_idx + 1
                seq_clip_idx = seq_clip_idx + 1

        Xch = np.asarray(Xch)
        print('Xch.shape: ', Xch.shape)
        print('len(ch_st_idxs): ', len(ch_st_idxs))
        print('fps set: {}'.format(fps_set))
    else:
        # Shape of input to AR model
        Xch = np.zeros((n_seqs, Nv, chunk_sz, raw_ft_dim))
        # Store each seq's chunks' start indices
        ch_st_idxs = np.zeros((n_seqs, Nv))

        # Chunk each seq.
        for seq_i_idx in range(n_seqs):
            T, _ = X[seq_i_idx].shape
            st_idxs = np.linspace(0, T - chunk_sz, Nv, dtype=int)
            ch_st_idxs[seq_i_idx] = st_idxs
            # Iteratively store each chunk
            for nv_j, st_idx in enumerate(st_idxs):
                Xch[seq_i_idx, nv_j] = X[seq_i_idx][st_idx: chunk_sz + st_idx]
    return Xch, ch_st_idxs

# chunking
chunk_X, info_dict = get_chunks(d_ntu['X'], d['fps'])
X = np.reshape(chunk_X, (chunk_X.shape[0], -1, chunk_X.shape[3]))
print('chunk_X.shape: ', chunk_X.shape)

sample_dur = 1.0  # 8.0/30.0
is_trans = False
spine_bone = np.array([0, 3])
hip_bone = np.array([2, 1])
# Display dataset samples stats
print('Samples = ', X.shape)

new_X = prep_x_for_norm(X, is_trans=is_trans)
print('Shape of prepped X: ', new_X.shape)

pre_X = preprocess.pre_normalization(new_X, skeleton_type='smpl')

print('len(pre_X): ', len(pre_X))
X_arr = np.array(pre_X)
print('X_arr.shape: ', X_arr.shape)
X_reshape = np.transpose(X_arr, (0, 2, 1, 3, 4))
X_reshape = np.reshape(X_reshape, (X_reshape.shape[0], X_reshape.shape[1], -1))
X_reshape = np.reshape(X_reshape, (X_reshape.shape[0], 100, X_reshape.shape[1] // 100, -1))
print('X_reshape.shape: ', X_reshape.shape)

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
           20: ['misc']
           }
print('# Action categories = ', len(idx2als))
al2idx = {k: idx for idx in idx2als for k in idx2als[idx]}
print('al2idx: ', al2idx)

skip_misc = False  # True
sk_type = 'smpl'  #'nturgbd'  # 'smpl'
num_f = 30
out_path = 'data/babel/'  # {0}f_smpl_sk_PC-AR_top20'.format(num_f)
os.makedirs('data/babel/', exist_ok=True)
version = 'hongwei'
np.save('{}babel_smpl_sk_Nv_100_csz_8_ip_to_AR_feat_ext_v{}.npy'.format(out_path, version), X_reshape)
print('out_path: ', out_path)

info_json_path = '{}info_v{}.json'.format(out_path, version)
with open(info_json_path, 'w') as f:
    json.dump(info_dict, f)
print('Save pred to {}'.format(info_json_path))
#
# rand_idxs = np.array(range(0, 10))  # np.random.randint(0, high=len(X_arr), size=10, dtype=int)
# for i, idx in enumerate(rand_idxs):
#     print('{1}f_nturgbd_ntu_prep_test_{0}_top20'.format(i, num_f))
#     # Sample the seq.
#     # x, act_cats, url = X_arr[idx], Y[idx], vids[idx]
#     x = X_arr[idx]
#     x = x[:, ::8]
#     print('x1 Data shape = {0}'.format(x.shape))
#     # pdb.set_trace()
#     # (C, T, V, 1) --> (T, V, C)
#     x = x[:, :, :, 0].transpose(1, 2, 0)
#     T = x.shape[0]
#     print('x2 Data shape = {0}'.format(x.shape))
#
#     if 'smpl' == sk_type:
#         x = np.concatenate((np.zeros((T, 3)), x.reshape(T, -1)), axis=-1)
#         print('Data shape = {0}'.format(x.shape))
#
#     print('# Seconds of video = ', x.shape[0] / 30.0)
#     # print('URL = ', url)
#     # print('Action cats idxs = ', act_cats)
#
#     if 'smpl' == sk_type:
#         viz.viz_seq(torch.Tensor(x[:, 0:69]), 'viz/{1}f_smpl_prep_test_{0}_top20'.format(i, num_f), \
#                     sk_type='smpl', debug=True)
#     else:
#         viz.viz_seq(torch.Tensor(x), 'viz/{1}f_nturgbd_ntu_prep_test_{0}_top20'.format(i, num_f), \
#                     sk_type='nturgbd', debug=True)
#     print('-' * 50)


