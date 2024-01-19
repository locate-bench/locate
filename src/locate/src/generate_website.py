import json
import os
import pdb
import pickle
import numpy as np
import torch

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
           # Large diversity expected
           10: ['stretch', 'yoga', 'exercise / training'],
           # Is this distinct enough from take / pick something up? Yeah, I think so.
           11: ['turn', 'spin'],
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


def generate_website(json_path='../../activitygraph_transformer_deformable_st_v3/output/checkpoints_/' +
                     'checkpoints_numqueries100_lr1e-5_lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned' +
                     '_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers4_hdim256_sr1_batchsize2_nposembdict512_numinputs100/95_babel_detection_result_agt.json',
                     pickle_path='../../activitygraph_transformer_deformable_st_v3/output/checkpoints_/' +
                     'checkpoints_numqueries100_lr1e-5_lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned' +
                     '_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers4_hdim256_sr1_batchsize2_nposembdict512_numinputs100/95_babel_groundtruth_result_agt.pkl',
                     website_template='../viz/viz_webpage_template.html', save_website='../viz/viz_webpage_saved.html',
                     npy_folder='../../activitygraph_transformer_deformable_st_v3/data/babel/i3d_feats/',
                     num_vid_vis=50,
                     babel_label_path='../../babel_tools/babel_v1.0_release'):
    train_json = os.path.join(babel_label_path, 'train.json')  # train_extra.json
    val_json = os.path.join(babel_label_path, 'val.json')  # val_extra.json

    print('Training Set')
    url_dur_mapping = {}

    with open(train_json, 'r') as f:
        train_label = json.load(f)
    for i, (k, v) in enumerate(train_label.items()):
        if i % 1000 == 0:
            print('Processing {} / {}'.format(i, len(train_label)))
        if v['url'] in url_dur_mapping.keys():
            print('url_dur_mapping[v[url]]: {}, v[dur]: {}, url_dur_mapping[v[url]]==v[dur]: {}'
            .format(url_dur_mapping[v['url']], v['dur'], url_dur_mapping[v['url']] == v['dur']))
        url_dur_mapping[v['url']] = v['dur']


    print('Val Set')
    with open(val_json, 'r') as f:
        val_label = json.load(f)
    for i, (k, v) in enumerate(val_label.items()):
        if i % 1000 == 0:
            print('Processing {} / {}'.format(i, len(val_label)))
        if v['url'] in url_dur_mapping.keys():
            print('url_dur_mapping[v[url]]: {}, v[dur]: {}, url_dur_mapping[v[url]]==v[dur]: {}'
                  .format(url_dur_mapping[v['url']], v['dur'], url_dur_mapping[v['url']] == v['dur'])
                  )
        url_dur_mapping[v['url']] = v['dur']
    # d['ann_idx'] = list(range(len(d['Psi'])))


    npy_file_list = os.listdir(npy_folder)
    with open(pickle_path, 'rb') as f:
        pickle_file = pickle.load(f)
    gt = pickle_file[0]
    # print(pickle_file[0]['video_test_0006626'])

    # raise NotImplementedError
    with open(json_path, 'r') as f:
        json_file = json.load(f)
    pred = json_file['results']
    selected_vid = list(pred.keys())[:num_vid_vis]
    viz_ds = {}
    for i, each_vid in enumerate(selected_vid):
        for each_npy in npy_file_list:
            if each_vid in each_npy:
                # 'video_validation_0002072_https:--babel-renders.s3.eu-central-1.amazonaws.com-010480.mp4@0002072.npy'
                url = each_npy.split('@')[0].split(
                    '_')[-1].replace('https:--', 'https://').replace('.com-', '.com/')
                break

        gt_vid = gt[each_vid]
        pred_vid = pred[each_vid]
        # print('each_vid, gt_vid: ', each_vid, gt_vid)
        pred_segments_list = []
        gt_segments_list = []
        for each_pred in pred_vid:
            # print('each_pred: ', each_pred)
            # print(each_pred['segment'], each_pred['label'])
            # print(idx2al[each_pred['label']])
            # print(idx2al)
            # print(each_pred['segment'][0], each_pred['segment'][1])
            pred_segments_list.append({'start': each_pred['segment'][0] / 100. * url_dur_mapping[url], 
                                       'end': each_pred['segment'][1] / 100. * url_dur_mapping[url],
                                       'action': idx2al[each_pred['label']], })
        for each_gt_segments, each_gt_labels in zip(gt_vid['segments'].detach().cpu().numpy(),
                                                    gt_vid['labels'].detach().cpu().numpy()):
            gt_segments_list.append({'start': each_gt_segments[0] * 100 / 100. * url_dur_mapping[url], 
                                     'end': each_gt_segments[1] * 100 / 100. * url_dur_mapping[url],
                                     'action': idx2al[each_gt_labels], })
        print(i, url, url_dur_mapping[url], 'each_gt_segments: {}, each_pred[segment]: {}'.format(
              each_gt_segments, each_pred['segment']))
        viz_ds[url] = {
            'index': i,
            'gt_segments': gt_segments_list,
            'pred_segments': pred_segments_list,
        }
        print('url, viz_ds[url]: ', url, viz_ds[url])

    webp = open(website_template).read()
    new_webp = webp.replace(r'var vid_act_map = {}',
                            'var vid_act_map = {0}'.format(viz_ds))
    wb_rname = 'saved'
    with open('../viz/{0}_{1}_viz.html'.format(wb_rname, '0'), 'w') as outf:
        outf.write(new_webp)

def compare_v3_v4(json_path_v3='../../activitygraph_transformer_deformable_st_v3/output/checkpoints_/checkpoints_numqueries100_lr1e-5_'
                  'lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers'
                  '4_hdim256_sr1_batchsize2_nposembdict512_numinputs100_gc_sc/95_babel_detection_val_agt.json',
             pickle_path_v3='../../activitygraph_transformer_deformable_st_v3/output/checkpoints_/checkpoints_numqueries100_lr1e-5_lrd'
                'rop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers4_h'
                'dim256_sr1_batchsize2_nposembdict512_numinputs100_gc_sc/95_babel_groundtruth_val_agt.pkl',
             json_path_v4='../../activitygraph_transformer_deformable_st_v4/output/checkpoints_/20210929-210918_checkpoints_numqueries100_lr1e-5_lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers4_hdim256_sr1_batchsize2_nposembdict512_numinputs100/95_babel_detection_val_agt_merge.json',
             pickle_path_v4='../../activitygraph_transformer_deformable_st_v4/output/checkpoints_/20210929-210918_checkpoints_numqueries100_lr1e-5_lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers4_hdim256_sr1_batchsize2_nposembdict512_numinputs100/95_babel_groundtruth_val_agt_merge.pkl',
             babel_label_path='../../babel_tools/babel_v1.0_release',
            save_name='compare_v3_v4',
            npy_folder='../../activitygraph_transformer_deformable_st_v4/data/babel/i3d_feats/',
            ):
    train_json = os.path.join(babel_label_path, 'train.json')  # train_extra.json
    val_json = os.path.join(babel_label_path, 'val.json')  # val_extra.json

    print('Training Set')
    url_dur_mapping = {}
    url_gtAction_mapping = {}

    with open(train_json, 'r') as f:
        train_label = json.load(f)
    for i, (k, v) in enumerate(train_label.items()):
        if i % 1000 == 0:
            print('Processing {} / {}'.format(i, len(train_label)))
        if v['url'] in url_dur_mapping.keys():
            print('url_dur_mapping[v[url]]: {}, v[dur]: {}, url_dur_mapping[v[url]]==v[dur]: {}'
                  .format(url_dur_mapping[v['url']], v['dur'], url_dur_mapping[v['url']] == v['dur']))
        url_dur_mapping[v['url']] = v['dur']
        segment_label = {'segments': [], 'labels': []}
        try:
            labels = v["frame_ann"]["labels"]
            for each_label in labels:
                act_cat = each_label['act_cat']
                for each_act_cat in act_cat:
                    start_t = each_label['start_t']
                    end_t = each_label['end_t']
                    for each_renamed_act in idx2al.values():
                        if each_act_cat in each_renamed_act:
                            each_act_cat = each_renamed_act
                            # print('each_act_cat: ', each_act_cat)
                    segment_label['segments'].append([start_t, end_t])
                    segment_label['labels'].append(each_act_cat)
            url_gtAction_mapping[v['url']] = segment_label
        except:
            labels = v['seq_ann']['labels']
            for each_label in labels:
                for each_act_cat in each_label['act_cat']:
                    start_t = 0.
                    end_t = v['dur']
                    for each_renamed_act in idx2al.values():
                        if each_act_cat in each_renamed_act:
                            each_act_cat = each_renamed_act
                            # print('each_act_cat: ', each_act_cat)
                    segment_label['segments'].append([start_t, end_t])
                    segment_label['labels'].append(each_act_cat)
            url_gtAction_mapping[v['url']] = segment_label

    print('Val Set')
    with open(val_json, 'r') as f:
        val_label = json.load(f)
    for i, (k, v) in enumerate(val_label.items()):
        if i % 1000 == 0:
            print('Processing {} / {}'.format(i, len(val_label)))
        if v['url'] in url_dur_mapping.keys():
            print('url_dur_mapping[v[url]]: {}, v[dur]: {}, url_dur_mapping[v[url]]==v[dur]: {}'
                  .format(url_dur_mapping[v['url']], v['dur'], url_dur_mapping[v['url']] == v['dur'])
                  )
        url_dur_mapping[v['url']] = v['dur']
        segment_label = {'segments': [], 'labels': []}
        try:
            labels = v["frame_ann"]["labels"]
            for each_label in labels:
                act_cat = each_label['act_cat']
                for each_act_cat in act_cat:
                    start_t = each_label['start_t']
                    end_t = each_label['end_t']
                    for each_renamed_act in idx2al.values():
                        if each_act_cat in each_renamed_act:
                            each_act_cat = each_renamed_act
                            # print('each_act_cat: ', each_act_cat)
                    segment_label['segments'].append([start_t, end_t])
                    segment_label['labels'].append(each_act_cat)
            url_gtAction_mapping[v['url']] = segment_label
        except:
            labels = v['seq_ann']['labels']
            for each_label in labels:
                for each_act_cat in each_label['act_cat']:
                    start_t = 0.
                    end_t = v['dur']
                    for each_renamed_act in idx2al.values():
                        if each_act_cat in each_renamed_act:
                            each_act_cat = each_renamed_act
                            # print('each_act_cat: ', each_act_cat)
                    segment_label['segments'].append([start_t, end_t])
                    segment_label['labels'].append(each_act_cat)
            url_gtAction_mapping[v['url']] = segment_label
    npy_file_list = os.listdir(npy_folder)
    with open(pickle_path_v3, 'rb') as f:
        pickle_file = pickle.load(f)
    gt = pickle_file[0]

    with open(pickle_path_v4, 'rb') as f:
        pickle_file_merge = pickle.load(f)
    gt_merge = pickle_file_merge[0]
    with open(json_path_v4, 'r') as f:
        json_file_merge = json.load(f)
    pred_merge = json_file_merge['results']

    selected_vid = list(pred_merge.keys())  # [:num_vid_vis]
    for i, each_vid in enumerate(selected_vid):
        for each_npy in npy_file_list:
            if each_vid in each_npy:
                # 'video_validation_0002072_https:--babel-renders.s3.eu-central-1.amazonaws.com-010480.mp4@0002072.npy'
                url = each_npy.split('@')[0].split(
                    '_')[-1].replace('https:--', 'https://').replace('.com-', '.com/')
                break

        gt_segments_list = []
        gt_segments_merge_list = []
        gt_vid = url_gtAction_mapping[url]
        for each_gt_segments, each_gt_labels in zip(gt_vid['segments'],
                                                    gt_vid['labels']):
            if each_gt_labels not in idx2al.values():
                continue
            gt_segments_list.append({'start': each_gt_segments[0],
                                     'end': each_gt_segments[1],
                                     'action': each_gt_labels,})
        # else:
        gt_merge_vid = gt_merge[each_vid]
        for each_gt_segments, each_gt_labels in zip(gt_merge_vid['segments'].detach().cpu().numpy(),
                                                    gt_merge_vid['labels'].detach().cpu().numpy()):
            gt_segments_merge_list.append({'start': each_gt_segments[0] * 100 / 100. * url_dur_mapping[url],
                                     'end': each_gt_segments[1] * 100 / 100. * url_dur_mapping[url],
                                     'action': idx2al[each_gt_labels], })

        # print('each_vid: {},\nurl: {},\neach_npy: {},\ngt_vid: {},\ngt_merge_vid: {}'
        #       .format(each_vid, url, each_npy, gt_segments_list, gt_segments_merge_list))
        print('each_npy: {},\ngt_vid: {},\ngt_merge_vid: {}'
              .format(each_npy, gt_segments_list, gt_segments_merge_list))
        if i > 20:
            raise NotImplementedError

    # wb_rname = 'saved'
    # save_name = '../viz/{}.txt'.format(save_name)
    # with open(save_name, 'w') as outf:
    #
    #     print('Website saved to {}'.format(save_name))

def generate_website_given_actions(json_path='../../activitygraph_transformer_deformable_st_v3/output/checkpoints_/checkpoints_numqueries100_lr1e-5_'
                  'lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers'
                  '4_hdim256_sr1_batchsize2_nposembdict512_numinputs100_gc_sc/95_babel_detection_val_agt.json',
                     pickle_path='../../activitygraph_transformer_deformable_st_v3/output/checkpoints_/checkpoints_numqueries100_lr1e-5_lrd'
                'rop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers4_h'
                'dim256_sr1_batchsize2_nposembdict512_numinputs100_gc_sc/95_babel_groundtruth_val_agt.pkl',
                     website_template='../viz/viz_webpage_template.html', save_website='../viz/viz_webpage_saved.html',
                     npy_folder='../../activitygraph_transformer_deformable_st_v3/data/babel/i3d_feats/',
                     num_vid_vis=None,
                     babel_label_path='../../babel_tools/babel_v1.0_release',
                    given_actions=None,
                    use_gt_all_classes=False,
                    save_name='viz_gt_unmerge',
                    debug=False):
    if given_actions is None:
        given_actions = ['kneel']
        # given_actions = ['hit or punch', 'grasp object', 'scratch or touching face or touching body parts']
    train_json = os.path.join(babel_label_path, 'train.json')  # train_extra.json
    val_json = os.path.join(babel_label_path, 'val.json')  # val_extra.json

    print('Training Set')
    url_dur_mapping = {}
    url_gtAction_mapping = {}

    with open(train_json, 'r') as f:
        train_label = json.load(f)
    for i, (k, v) in enumerate(train_label.items()):
        if i % 1000 == 0:
            print('Processing {} / {}'.format(i, len(train_label)))
        if v['url'] in url_dur_mapping.keys():
            print('url_dur_mapping[v[url]]: {}, v[dur]: {}, url_dur_mapping[v[url]]==v[dur]: {}'
            .format(url_dur_mapping[v['url']], v['dur'], url_dur_mapping[v['url']] == v['dur']))
        url_dur_mapping[v['url']] = v['dur']
        segment_label = {'segments': [], 'labels': []}
        try:
            labels = v["frame_ann"]["labels"]
            for each_label in labels:
                act_cat = each_label['act_cat']
                for each_act_cat in act_cat:
                    start_t = each_label['start_t']
                    end_t = each_label['end_t']
                    for each_renamed_act in idx2al.values():
                        if each_act_cat in each_renamed_act:
                            each_act_cat = each_renamed_act
                            # print('each_act_cat: ', each_act_cat)
                    segment_label['segments'].append([start_t, end_t])
                    segment_label['labels'].append(each_act_cat)
            url_gtAction_mapping[v['url']] = segment_label
        except:
            labels = v['seq_ann']['labels']
            for each_label in labels:
                for each_act_cat in each_label['act_cat']:
                    start_t = 0.
                    end_t = v['dur']
                    for each_renamed_act in idx2al.values():
                        if each_act_cat in each_renamed_act:
                            each_act_cat = each_renamed_act
                            # print('each_act_cat: ', each_act_cat)
                    segment_label['segments'].append([start_t, end_t])
                    segment_label['labels'].append(each_act_cat)
            url_gtAction_mapping[v['url']] = segment_label

    print('Val Set')
    with open(val_json, 'r') as f:
        val_label = json.load(f)
    for i, (k, v) in enumerate(val_label.items()):
        if i % 1000 == 0:
            print('Processing {} / {}'.format(i, len(val_label)))
        if v['url'] in url_dur_mapping.keys():
            print('url_dur_mapping[v[url]]: {}, v[dur]: {}, url_dur_mapping[v[url]]==v[dur]: {}'
                  .format(url_dur_mapping[v['url']], v['dur'], url_dur_mapping[v['url']] == v['dur'])
                  )
        url_dur_mapping[v['url']] = v['dur']
        segment_label = {'segments': [], 'labels': []}
        try:
            labels = v["frame_ann"]["labels"]
            for each_label in labels:
                act_cat = each_label['act_cat']
                for each_act_cat in act_cat:
                    start_t = each_label['start_t']
                    end_t = each_label['end_t']
                    for each_renamed_act in idx2al.values():
                        if each_act_cat in each_renamed_act:
                            each_act_cat = each_renamed_act
                            # print('each_act_cat: ', each_act_cat)
                    segment_label['segments'].append([start_t, end_t])
                    segment_label['labels'].append(each_act_cat)
            url_gtAction_mapping[v['url']] = segment_label
        except:
            labels = v['seq_ann']['labels']
            for each_label in labels:
                for each_act_cat in each_label['act_cat']:
                    start_t = 0.
                    end_t = v['dur']
                    for each_renamed_act in idx2al.values():
                        if each_act_cat in each_renamed_act:
                            each_act_cat = each_renamed_act
                            # print('each_act_cat: ', each_act_cat)
                    segment_label['segments'].append([start_t, end_t])
                    segment_label['labels'].append(each_act_cat)
            url_gtAction_mapping[v['url']] = segment_label

    npy_file_list = os.listdir(npy_folder)
    with open(pickle_path, 'rb') as f:
        pickle_file = pickle.load(f)
    gt = pickle_file[0]
    # print(pickle_file[0]['video_test_0006626'])

    # raise NotImplementedError
    with open(json_path, 'r') as f:
        json_file = json.load(f)
    pred = json_file['results']
    selected_vid = list(pred.keys())  # [:num_vid_vis]
    viz_ds = {}
    viz_count = 0
    for i, each_vid in enumerate(selected_vid):
        for each_npy in npy_file_list:
            if each_vid in each_npy:
                # 'video_validation_0002072_https:--babel-renders.s3.eu-central-1.amazonaws.com-010480.mp4@0002072.npy'
                # url = each_npy.split('@')[0].split(
                #     '_')[-1].replace('https:--', 'https://').replace('.com-', '.com/')
                url_original = each_npy.split('@')[0].split('_')[-1]
                url = url_original.replace('https:--', 'https://').replace('.com-', '.com/')
                break

        pred_vid = pred[each_vid]
        # print('each_vid, gt_vid: ', each_vid, gt_vid)
        pred_segments_list = []
        gt_segments_list = []
        skip_flag = True
        for each_pred in pred_vid:
            pred_segments_list.append({'start': each_pred['segment'][0] / 100. * url_dur_mapping[url],
                                       'end': each_pred['segment'][1] / 100. * url_dur_mapping[url],
                                       'action': idx2al[each_pred['label']], })
            # if idx2al[each_pred['label']] in given_actions:
            #     skip_flag = False
        if use_gt_all_classes:
            # print('each_vid: ', each_vid)  # video_test_0006615
            gt_vid = url_gtAction_mapping[url]
            for each_gt_segments, each_gt_labels in zip(gt_vid['segments'],
                                                        gt_vid['labels']):
                gt_segments_list.append({'start': each_gt_segments[0],
                                         'end': each_gt_segments[1],
                                         'action': each_gt_labels,})
                if each_gt_labels in given_actions:
                    skip_flag = False
        else:
            gt_vid = gt[each_vid]
            for each_gt_segments, each_gt_labels in zip(gt_vid['segments'].detach().cpu().numpy(),
                                                        gt_vid['labels'].detach().cpu().numpy()):
                gt_segments_list.append({'start': each_gt_segments[0] * 100 / 100. * url_dur_mapping[url],
                                         'end': each_gt_segments[1] * 100 / 100. * url_dur_mapping[url],
                                         'action': idx2al[each_gt_labels], })
                if idx2al[each_gt_labels] in given_actions:
                    skip_flag = False
        if num_vid_vis is not None and viz_count >= num_vid_vis:
            break
        elif skip_flag:
            continue
        print(i, url, url_dur_mapping[url], 'each_gt_segments: {}, each_pred[labels]: {}'.format(
              each_gt_segments, each_gt_labels))
        viz_ds[url] = {
            'index': i,
            'gt_segments': gt_segments_list,
            'pred_segments': pred_segments_list,
        }
        print('url, viz_ds[url]: ', url, viz_ds[url])
        viz_count = viz_count + 1
        if debug and i == 103:  # url in ['https://babel-renders.s3.eu-central-1.amazonaws.com/008843.mp4']:
            print('url: {}'.format(url))
            #  and each_vid in 'video_test_0007697_https:--babel-renders.s3.eu-central-1.amazonaws.com-006809.mp4@0007697.npy':
            print('gt_vid[segments]: {}, gt_vid[labels]: {}'.format(gt_vid['segments'], gt_vid['labels']))
            print('gt_segments_list: {}'.format(gt_segments_list))
            print('url_dur_mapping[url]: {}'.format(url_dur_mapping[url]))
            print('url_gtAction_mapping[url]: {}'.format(url_gtAction_mapping[url]))
            pdb.set_trace()

    webp = open(website_template).read()
    new_webp = webp.replace(r'var vid_act_map = {}',
                            'var vid_act_map = {0}'.format(viz_ds))
    wb_rname = 'saved'
    save_name = '../viz/{}.html'.format(save_name)
    with open(save_name, 'w') as outf:
        outf.write(new_webp)
        print('Website saved to {}'.format(save_name))

def generate_website_hongwei(json_path='../../activitygraph_transformer_deformable_st_v3/output/checkpoints_/checkpoints_numqueries100_lr1e-5_'
                  'lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers'
                  '4_hdim256_sr1_batchsize2_nposembdict512_numinputs100_gc_sc/95_babel_detection_val_agt.json',
                     pickle_path='../../activitygraph_transformer_deformable_st_v3/output/checkpoints_/checkpoints_numqueries100_lr1e-5_lrd'
                'rop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers4_h'
                'dim256_sr1_batchsize2_nposembdict512_numinputs100_gc_sc/95_babel_groundtruth_val_agt.pkl',
                     website_template='../3D-TAL/viz/viz_webpage_template_human.html', save_website='../viz/viz_webpage_saved.html',
                     npy_folder='data/babel/i3d_feats/',
                     num_vid_vis=None,
                     babel_label_path='data/hongwei',
                    given_actions=None,
                    use_gt_all_classes=False,
                    save_name='viz_gt_unmerge',
                    debug=False):
    # if given_actions is None:
    #     given_actions = ['kneel']
        # given_actions = ['hit or punch', 'grasp object', 'scratch or touching face or touching body parts']
    train_json = os.path.join(babel_label_path, 'train.json')  # train_extra.json
    val_json = os.path.join(babel_label_path, 'val.json')  # val_extra.json

    print('Training Set')
    url_dur_mapping = {}
    url_gtAction_mapping = {}

    with open(train_json, 'r') as f:
        train_label = json.load(f)
    for i, (k, v) in enumerate(train_label.items()):
        if i % 1000 == 0:
            print('Processing {} / {}'.format(i, len(train_label)))
        if v['url'] in url_dur_mapping.keys():
            print('url_dur_mapping[v[url]]: {}, v[dur]: {}, url_dur_mapping[v[url]]==v[dur]: {}'
            .format(url_dur_mapping[v['url']], v['dur'], url_dur_mapping[v['url']] == v['dur']))
        url_dur_mapping[v['url']] = v['dur']
        segment_label = {'segments': [], 'labels': []}
        try:
            labels = v["frame_ann"]["labels"]
            for each_label in labels:
                act_cat = each_label['act_cat']
                for each_act_cat in act_cat:
                    start_t = each_label['start_t']
                    end_t = each_label['end_t']
                    for each_renamed_act in idx2al.values():
                        if each_act_cat in each_renamed_act:
                            each_act_cat = each_renamed_act
                            # print('each_act_cat: ', each_act_cat)
                    segment_label['segments'].append([start_t, end_t])
                    segment_label['labels'].append(each_act_cat)
            url_gtAction_mapping[v['url']] = segment_label
        except:
            labels = v['seq_ann']['labels']
            for each_label in labels:
                for each_act_cat in each_label['act_cat']:
                    start_t = 0.
                    end_t = v['dur']
                    for each_renamed_act in idx2al.values():
                        if each_act_cat in each_renamed_act:
                            each_act_cat = each_renamed_act
                            # print('each_act_cat: ', each_act_cat)
                    segment_label['segments'].append([start_t, end_t])
                    segment_label['labels'].append(each_act_cat)
            url_gtAction_mapping[v['url']] = segment_label

    print('Val Set')
    with open(val_json, 'r') as f:
        val_label = json.load(f)
    for i, (k, v) in enumerate(val_label.items()):
        if i % 1000 == 0:
            print('Processing {} / {}'.format(i, len(val_label)))
        if v['url'] in url_dur_mapping.keys():
            print('url_dur_mapping[v[url]]: {}, v[dur]: {}, url_dur_mapping[v[url]]==v[dur]: {}'
                  .format(url_dur_mapping[v['url']], v['dur'], url_dur_mapping[v['url']] == v['dur'])
                  )
        url_dur_mapping[v['url']] = v['dur']
        segment_label = {'segments': [], 'labels': []}
        try:
            labels = v["frame_ann"]["labels"]
            for each_label in labels:
                act_cat = each_label['act_cat']
                for each_act_cat in act_cat:
                    start_t = each_label['start_t']
                    end_t = each_label['end_t']
                    for each_renamed_act in idx2al.values():
                        if each_act_cat in each_renamed_act:
                            each_act_cat = each_renamed_act
                            # print('each_act_cat: ', each_act_cat)
                    segment_label['segments'].append([start_t, end_t])
                    segment_label['labels'].append(each_act_cat)
            url_gtAction_mapping[v['url']] = segment_label
        except:
            labels = v['seq_ann']['labels']
            for each_label in labels:
                for each_act_cat in each_label['act_cat']:
                    start_t = 0.
                    end_t = v['dur']
                    for each_renamed_act in idx2al.values():
                        if each_act_cat in each_renamed_act:
                            each_act_cat = each_renamed_act
                            # print('each_act_cat: ', each_act_cat)
                    segment_label['segments'].append([start_t, end_t])
                    segment_label['labels'].append(each_act_cat)
            url_gtAction_mapping[v['url']] = segment_label

    npy_file_list = os.listdir(npy_folder)
    with open(pickle_path, 'rb') as f:
        pickle_file = pickle.load(f)
    gt = pickle_file[0]
    # print(pickle_file[0]['video_test_0006626'])

    # raise NotImplementedError
    with open(json_path, 'r') as f:
        json_file = json.load(f)
    pred = json_file['results']
    selected_vid = list(pred.keys())  # [:num_vid_vis]
    viz_ds = {}
    viz_count = 0
    for i, each_vid in enumerate(selected_vid):
        for each_npy in npy_file_list:
            if each_vid in each_npy:
                # 'video_validation_0002072_https:--babel-renders.s3.eu-central-1.amazonaws.com-010480.mp4@0002072.npy'
                # url = each_npy.split('@')[0].split(
                #     '_')[-1].replace('https:--', 'https://').replace('.com-', '.com/')
                url_original = each_npy.split('@')[0].split('_')[-1]
                url = url_original.replace('https:--', 'https://').replace('.com-', '.com/')
                break

        pred_vid = pred[each_vid]
        # print('each_vid, gt_vid: ', each_vid, gt_vid)
        pred_segments_list = []
        gt_segments_list = []
        skip_flag = True
        for each_pred in pred_vid:
            pred_segments_list.append({'start': each_pred['segment'][0] / 100. * url_dur_mapping[url],
                                       'end': each_pred['segment'][1] / 100. * url_dur_mapping[url],
                                       'action': idx2al[each_pred['label']], })
            # if idx2al[each_pred['label']] in given_actions:
            #     skip_flag = False
        if use_gt_all_classes:
            # print('each_vid: ', each_vid)  # video_test_0006615
            gt_vid = url_gtAction_mapping[url]
            for each_gt_segments, each_gt_labels in zip(gt_vid['segments'],
                                                        gt_vid['labels']):
                gt_segments_list.append({'start': each_gt_segments[0],
                                         'end': each_gt_segments[1],
                                         'action': each_gt_labels,})
                if given_actions is None or each_gt_labels in given_actions:
                    skip_flag = False
        else:
            gt_vid = gt[each_vid]
            for each_gt_segments, each_gt_labels in zip(gt_vid['segments'].detach().cpu().numpy(),
                                                        gt_vid['labels'].detach().cpu().numpy()):
                gt_segments_list.append({'start': each_gt_segments[0] * 100 / 100. * url_dur_mapping[url],
                                         'end': each_gt_segments[1] * 100 / 100. * url_dur_mapping[url],
                                         'action': idx2al[each_gt_labels], })
                if given_actions is None or idx2al[each_gt_labels] in given_actions:
                    skip_flag = False
        if num_vid_vis is not None and viz_count >= num_vid_vis:
            break
        elif skip_flag:
            continue
        print(i, url, url_dur_mapping[url], 'each_gt_segments: {}, each_pred[labels]: {}'.format(
              each_gt_segments, each_gt_labels))
        viz_ds[url] = {
            'index': i,
            'gt_segments': gt_segments_list,
            'pred_segments': pred_segments_list,
        }
        print('url, viz_ds[url]: ', url, viz_ds[url])
        viz_count = viz_count + 1
        if debug and i == 103:  # url in ['https://babel-renders.s3.eu-central-1.amazonaws.com/008843.mp4']:
            print('url: {}'.format(url))
            #  and each_vid in 'video_test_0007697_https:--babel-renders.s3.eu-central-1.amazonaws.com-006809.mp4@0007697.npy':
            print('gt_vid[segments]: {}, gt_vid[labels]: {}'.format(gt_vid['segments'], gt_vid['labels']))
            print('gt_segments_list: {}'.format(gt_segments_list))
            print('url_dur_mapping[url]: {}'.format(url_dur_mapping[url]))
            print('url_gtAction_mapping[url]: {}'.format(url_gtAction_mapping[url]))
            pdb.set_trace()

    webp = open(website_template).read()
    new_webp = webp.replace(r'var vid_act_map = {}',
                            'var vid_act_map = {0}'.format(viz_ds))
    wb_rname = 'saved'
    os.makedirs('viz', exist_ok=True)
    save_name = 'viz/{}.html'.format(save_name)
    with open(save_name, 'w') as outf:
        outf.write(new_webp)
        print('Website saved to {}'.format(save_name))

def merge_v4_to_v3(json_path='../../activitygraph_transformer_deformable_st_v4/output/checkpoints_/20210929-210918_checkpoints_numqueries100_lr1e-5_lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers4_hdim256_sr1_batchsize2_nposembdict512_numinputs100/95_babel_detection_val_agt.json',
                   pickle_path='../../activitygraph_transformer_deformable_st_v4/output/checkpoints_/20210929-210918_checkpoints_numqueries100_lr1e-5_lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers4_hdim256_sr1_batchsize2_nposembdict512_numinputs100/95_babel_groundtruth_val_agt.pkl',
                   merge_json_path='../../activitygraph_transformer_deformable_st_v4/output/checkpoints_/20210929-210918_checkpoints_numqueries100_lr1e-5_lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers4_hdim256_sr1_batchsize2_nposembdict512_numinputs100/95_babel_detection_val_agt_merge.json',
                   merge_pickle_path='../../activitygraph_transformer_deformable_st_v4/output/checkpoints_/20210929-210918_checkpoints_numqueries100_lr1e-5_lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers4_hdim256_sr1_batchsize2_nposembdict512_numinputs100/95_babel_groundtruth_val_agt_merge.pkl',
                   website_template='../viz/viz_webpage_template.html', save_website='../viz/viz_webpage_saved.html',
                   npy_folder='../../activitygraph_transformer_deformable_st_v4/data/babel/i3d_feats/',
                   thres=5,
                   thres_gt=5,
                   num_vid_vis=None,
                   babel_label_path='../../babel_tools/babel_v1.0_release',
                   given_actions=None,
                   use_gt_all_classes=True,
                   version='v4',
                   skeleton_dir='../../tools/normalization_vis/',
                   npz_prefix='../../Dataset/amass_unzip_smpl_jpos',
                   amass_unzip_prefix='../../Dataset/amass_unzip',
                   num_chunks_per_sequence=100,
                   chunk_sz=8,
                   debug=True,
                   ):
    train_json = os.path.join(babel_label_path, 'train.json')  # train_extra.json
    val_json = os.path.join(babel_label_path, 'val.json')  # val_extra.json

    print('Training Set')
    url_dur_mapping = {}
    url_gtAction_mapping = {}

    with open(train_json, 'r') as f:
        train_label = json.load(f)
    for i, (k, v) in enumerate(train_label.items()):
        if i % 1000 == 0:
            print('Processing {} / {}'.format(i, len(train_label)))
        if v['url'] in url_dur_mapping.keys():
            print('url_dur_mapping[v[url]]: {}, v[dur]: {}, url_dur_mapping[v[url]]==v[dur]: {}'
                  .format(url_dur_mapping[v['url']], v['dur'], url_dur_mapping[v['url']] == v['dur']))
        url_dur_mapping[v['url']] = v['dur']
        segment_label = {'segments': [], 'labels': []}
        try:
            labels = v["frame_ann"]["labels"]
            for each_label in labels:
                act_cat = each_label['act_cat']
                for each_act_cat in act_cat:
                    start_t = each_label['start_t']
                    end_t = each_label['end_t']
                    for each_renamed_act in idx2al.values():
                        if each_act_cat in each_renamed_act:
                            each_act_cat = each_renamed_act
                            # print('each_act_cat: ', each_act_cat)
                    segment_label['segments'].append([start_t, end_t])
                    segment_label['labels'].append(each_act_cat)
            url_gtAction_mapping[v['url']] = segment_label
        except:
            labels = v['seq_ann']['labels']
            for each_label in labels:
                for each_act_cat in each_label['act_cat']:
                    start_t = 0.
                    end_t = v['dur']
                    for each_renamed_act in idx2al.values():
                        if each_act_cat in each_renamed_act:
                            each_act_cat = each_renamed_act
                            # print('each_act_cat: ', each_act_cat)
                    segment_label['segments'].append([start_t, end_t])
                    segment_label['labels'].append(each_act_cat)
            url_gtAction_mapping[v['url']] = segment_label

    print('Val Set')
    with open(val_json, 'r') as f:
        val_label = json.load(f)
    for i, (k, v) in enumerate(val_label.items()):
        if i % 1000 == 0:
            print('Processing {} / {}'.format(i, len(val_label)))
        if v['url'] in url_dur_mapping.keys():
            print('url_dur_mapping[v[url]]: {}, v[dur]: {}, url_dur_mapping[v[url]]==v[dur]: {}'
                  .format(url_dur_mapping[v['url']], v['dur'], url_dur_mapping[v['url']] == v['dur'])
                  )
        url_dur_mapping[v['url']] = v['dur']
        segment_label = {'segments': [], 'labels': []}
        try:
            labels = v["frame_ann"]["labels"]
            for each_label in labels:
                act_cat = each_label['act_cat']
                for each_act_cat in act_cat:
                    start_t = each_label['start_t']
                    end_t = each_label['end_t']
                    for each_renamed_act in idx2al.values():
                        if each_act_cat in each_renamed_act:
                            each_act_cat = each_renamed_act
                            # print('each_act_cat: ', each_act_cat)
                    segment_label['segments'].append([start_t, end_t])
                    segment_label['labels'].append(each_act_cat)
            url_gtAction_mapping[v['url']] = segment_label
        except:
            labels = v['seq_ann']['labels']
            for each_label in labels:
                for each_act_cat in each_label['act_cat']:
                    start_t = 0.
                    end_t = v['dur']
                    for each_renamed_act in idx2al.values():
                        if each_act_cat in each_renamed_act:
                            each_act_cat = each_renamed_act
                            # print('each_act_cat: ', each_act_cat)
                    segment_label['segments'].append([start_t, end_t])
                    segment_label['labels'].append(each_act_cat)
            url_gtAction_mapping[v['url']] = segment_label

    npy_file_list = os.listdir(npy_folder)
    with open(pickle_path, 'rb') as f:
        pickle_file = pickle.load(f)
    gt = pickle_file[0]
    gt_merge_dict = {}
    with open('../../data/babel/info_{}.json'.format(version), 'r') as f:
        info_dict = json.load(f)
    with open(json_path, 'r') as f:
        json_file = json.load(f)
    pred = json_file['results']
    pred_merge_dict = {}
    selected_vid = list(pred.keys())  # [:num_vid_vis]
    vid_i = 0
    while vid_i < len(selected_vid):
        each_vid = selected_vid[vid_i]
        for npy_i, each_npy in enumerate(npy_file_list):
            if each_vid in each_npy:
                # 'video_validation_0002072_https:--babel-renders.s3.eu-central-1.amazonaws.com-010480.mp4@0002072.npy'
                url_original = each_npy.split('@')[0].split('_')[-1]
                url = url_original.replace('https:--', 'https://').replace('.com-', '.com/')
                ann_idx_array = os.path.splitext(each_npy.split('@')[1])[0]
                ann_idx_array_int = int(ann_idx_array)
                ann_idx_array_str = str(ann_idx_array_int)
                break
        pred_vid = pred[each_vid]
        if not debug:
            print('each_vid: {}'.format(each_vid))
            print('each_npy: {}, npy_i: {}'.format(each_npy, npy_i))
            print('url: {}, ann_idx_array_int: {}'.format(url, ann_idx_array_int))
            # print('len(pred_segments_list): {}, pred_segments_list[0]: {}'.format(len(pred_segments_list), pred_segments_list[0]))
        # len(pred_segments_list): 1, pred_segments_list[0]: {'score': 0.16034483909606934, 'label': 12, 'segment': [0.029045434668660164, 29.258127212524414]}
        gt_vid = gt[each_vid]

        # print('gt_segments_list.keys(): ', gt_segments_list.keys())
        # gt_segments_list.keys():  dict_keys(['video_id', 'segments', 'labels', 'num_feat', 'length', 'sample_rate', 'fps', 'num_classes'])
        # print('info_dict.keys(): ', info_dict.keys())
        # max_frame_acc = info_dict[ann_idx_array_str][1]
        max_frame_acc = info_dict[ann_idx_array_str][2]
        num_clip = info_dict[ann_idx_array_str][3]
        prev_num_clip = num_clip
        if num_clip > 0:
            gt_vid['segments'] = gt_vid['segments'] + num_clip
            for pred_vid_i, each_pred in enumerate(pred_vid):
                each_pred['segment'][1] = each_pred['segment'][1] + num_chunks_per_sequence * num_clip
                each_pred['segment'][0] = each_pred['segment'][0] + num_chunks_per_sequence * num_clip
        gt_segments_list = gt_vid
        pred_segments_list = pred_vid
        if debug:
            print('1. vid_i: {}, max_frame: {}, info_dict[ann_idx_array_int]: {}'.format(vid_i, max_frame_acc, info_dict[ann_idx_array_str]))
        # print('url: {}, url_original: {}, npy_file_list[npy_i+1]: {}'.format(url, url_original, npy_file_list[npy_i+1]))
        while (npy_i+1 < len(npy_file_list)) and (url_original in npy_file_list[npy_i+1]):
            each_npy = npy_file_list[npy_i+1]
            ann_idx_array = os.path.splitext(each_npy.split('@')[1])[0]
            ann_idx_array_int = int(ann_idx_array)
            ann_idx_array_str = str(ann_idx_array_int)
            # max_frame_acc = max_frame_acc + info_dict[ann_idx_array_str][1]
            if debug:
                print('2. vid_i: {}, max_frame: {}, info_dict[ann_idx_array_int]: {}, ann_idx_array_str: {}'
                      .format(vid_i, max_frame_acc, info_dict[ann_idx_array_str], ann_idx_array_str))
            vid_i = vid_i + 1
            num_clip = info_dict[ann_idx_array_str][3]
            each_vid = selected_vid[vid_i]
            pred_vid = pred[each_vid]
            gt_vid = gt[each_vid]
            pred_segments_list_len = len(pred_segments_list)
            for pred_vid_i, each_pred in enumerate(pred_vid):
                find_flag = False
                for pred_segments_list_i in range(pred_segments_list_len):
                    each_base_pred = pred_segments_list[pred_segments_list_i]
                    # print('pred_vid_i: {}, pred_segments_list_i: {}'.format(pred_vid_i, pred_segments_list_i))
                    if num_clip - prev_num_clip == 1 and each_pred['label'] == each_base_pred['label'] and each_pred['segment'][0] < thres \
                            and each_base_pred['segment'][1] > num_chunks_per_sequence - thres:
                        find_flag = True
                        each_base_pred['segment'][1] = each_pred['segment'][1] + num_chunks_per_sequence * num_clip
                # for ... else ...
                # https://stackoverflow.com/questions/9979970/why-does-python-use-else-after-for-and-while-loops
                if not find_flag:
                    pred_segments_list.append({'label': each_pred['label'],
                                               'segment': [each_pred['segment'][0] + num_chunks_per_sequence * num_clip,
                                                           each_pred['segment'][1] + num_chunks_per_sequence * num_clip]})
            gt_segments_list_len = len(gt_segments_list['labels'])
            for each_gt_segments, each_gt_labels in zip(gt_vid['segments'], gt_vid['labels']):
                find_flag = False
                for each_base_gt_labels_idx in range(gt_segments_list_len):
                    each_base_gt_labels = gt_segments_list['labels'][each_base_gt_labels_idx]
                    # print('each_base_gt_labels_idx: {}, each_gt_segments: {}, gt_segments_list[segments][each_base_gt_labels_idx]: {}'
                    #       .format(each_base_gt_labels_idx, each_gt_segments, gt_segments_list['segments'][each_base_gt_labels_idx]))
                    # print('each_gt_labels: {} == each_base_gt_labels: {}, each_gt_labels == each_base_gt_labels: {}'
                    #       .format(each_gt_labels, each_base_gt_labels, each_gt_labels == each_base_gt_labels))
                    # each_gt_labels: 13 == each_base_gt_labels: 1, each_gt_labels == each_base_gt_labels
                    if num_clip - prev_num_clip == 1 and each_gt_labels == each_base_gt_labels and each_gt_segments[0] < thres_gt / num_chunks_per_sequence \
                            and gt_segments_list['segments'][each_base_gt_labels_idx][1] > 1 - thres_gt / num_chunks_per_sequence:
                        find_flag = True
                        gt_segments_list['segments'][each_base_gt_labels_idx][1] = each_gt_segments[1] + num_clip
                # for ... else ...
                # https://stackoverflow.com/questions/9979970/why-does-python-use-else-after-for-and-while-loops
                if not find_flag:
                    # print('1. gt_segments_list[labels]: {}, each_gt_labels: {}'
                    #       .format(gt_segments_list['labels'], each_gt_labels.unsqueeze(0)))
                    # print('2. gt_segments_list[segments]: {}, each_gt_segments + num_clip: {}'
                    #     .format(gt_segments_list['segments'], each_gt_segments.unsqueeze(0) + num_clip))
                    gt_segments_list['labels'] = torch.cat((gt_segments_list['labels'], each_gt_labels.unsqueeze(0)), dim=0)
                    gt_segments_list['segments'] = torch.cat((gt_segments_list['segments'], each_gt_segments.unsqueeze(0) + num_clip), dim=0)
                    # print('1. gt_segments_list[labels]: {}, each_gt_labels: {}'
                    #       .format(gt_segments_list['labels'], each_gt_labels.unsqueeze(0)))
                    # print('2. gt_segments_list[segments]: {}, each_gt_segments + num_clip: {}'
                    #     .format(gt_segments_list['segments'], each_gt_segments.unsqueeze(0) + num_clip))
            npy_i = npy_i + 1
            prev_num_clip = num_clip
        # update video index
        vid_i = vid_i + 1
        seq_dur = url_dur_mapping[url] * num_chunks_per_sequence * chunk_sz / max_frame_acc
        gt_segments_list_norm = {
            'labels': gt_segments_list['labels'],
            # 'segments': gt_segments_list['segments'] / (num_clip + (info_dict[ann_idx_array_str][1] / 8) / num_chunks_per_sequence),
            'segments': gt_segments_list['segments'] * seq_dur / url_dur_mapping[url],
        }
        pred_segments_list_norm = []
        for item in pred_segments_list:
            pred_segments_list_norm.append({'label': item['label'],
                                            'segment': [item['segment'][0] / (max_frame_acc / chunk_sz) * num_chunks_per_sequence,
                                                        item['segment'][1] / (max_frame_acc / chunk_sz) * num_chunks_per_sequence]})
            # 0 - num_chunks_per_sequence (100)
        # normalize the start end to 0-100, 0-1
        gt_merge_dict[each_vid] = gt_segments_list_norm
        pred_merge_dict[each_vid] = pred_segments_list_norm
        print('each_vid: {}'.format(each_vid))
        if debug and url in ['https://babel-renders.s3.eu-central-1.amazonaws.com/008843.mp4']:
            #  and each_vid in 'video_test_0007697_https:--babel-renders.s3.eu-central-1.amazonaws.com-006809.mp4@0007697.npy':
            print('gt_vid[segments]: {}, gt_vid[labels]: {}'.format(gt_vid['segments'], gt_vid['labels']))
            print('gt_merge_dict[each_vid]: {}'.format(gt_merge_dict[each_vid]))
            print('gt_segments_list: {}'.format(gt_segments_list))
            print('url_dur_mapping[url]: {}'.format(url_dur_mapping[url]))
            print('url_dur_mapping[url] * gt_merge_dict[each_vid][segments]: {}'
                  .format(url_dur_mapping[url] * gt_merge_dict[each_vid]['segments']))
            print('url_gtAction_mapping[url]: {}'.format(url_gtAction_mapping[url]))
            # print('(num_clip+1): {}'.format((num_clip+1)))
            # print('info_dict[ann_idx_array_str][1]: {}， '
            #       '(info_dict[ann_idx_array_str][1] / 8) / num_chunks_per_sequence: {}, '
            #       '(num_clip + (info_dict[ann_idx_array_str][1] / 8) / num_chunks_per_sequence): {}'
            #       .format(info_dict[ann_idx_array_str][1],
            #        (info_dict[ann_idx_array_str][1] / 8) / num_chunks_per_sequence,
            #        (num_clip + (info_dict[ann_idx_array_str][1] / 8) / num_chunks_per_sequence)))
            pdb.set_trace()

    if not debug:
        with open(merge_pickle_path, 'wb') as f:
            pickle.dump([gt_merge_dict], f)
        print('Save gt to {}'.format(merge_pickle_path))

        with open(merge_json_path, 'w') as f:
            json.dump({'results': pred_merge_dict}, f)
        print('Save pred to {}'.format(merge_json_path))

if __name__ == "__main__":
    # generate_website()
    mode = 'eval_vis'  # 'merge' # comp, gen
    if mode == 'comp':
        compare_v3_v4()
    elif mode == 'gen_unmerge':
        generate_website_given_actions()
    elif mode == 'gen_merge':
        generate_website_given_actions(json_path='../../activitygraph_transformer_deformable_st_v4/output/checkpoints_/20210929-210918_checkpoints_numqueries100_lr1e-5_lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers4_hdim256_sr1_batchsize2_nposembdict512_numinputs100/95_babel_detection_val_agt_merge.json',
                       pickle_path='../../activitygraph_transformer_deformable_st_v4/output/checkpoints_/20210929-210918_checkpoints_numqueries100_lr1e-5_lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers4_hdim256_sr1_batchsize2_nposembdict512_numinputs100/95_babel_groundtruth_val_agt_merge.pkl',
                       npy_folder='../../activitygraph_transformer_deformable_st_v4/data/babel/i3d_feats/',
                       save_name='viz_gt_merge',
                       )
    elif mode == 'eval_vis':
        latest_his = sorted(os.listdir(os.path.join('output', 'checkpoints_')))[-1]
        print('latest his: {}'.format(latest_his))
        generate_website_hongwei(
            json_path='output/checkpoints_/{}/00_babel_detection_val_agt.json'.format(latest_his),
            pickle_path='output/checkpoints_/{}/00_babel_groundtruth_val_agt.pkl'.format(latest_his),
            npy_folder='data/babel/i3d_feats/',
            save_name='viz_hongwei',
            )
    else: # merge
        merge_v4_to_v3()
