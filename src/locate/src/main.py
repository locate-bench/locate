import os
import argparse
import random
import numpy as np
import time
from pathlib import Path
import json
import datetime
import pickle
import torch
from torch.utils.data import DataLoader, DistributedSampler

## data loader
# import datasets
from datasets import build_dataset

## model training and utils
import utils.misc as utils
from models import build_model
from engine import train_one_epoch, evaluate
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
import wandb
os.environ["WANDB_API_KEY"] = "b00b2711e75723b6df804b383842ad17c46a84b0"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,default="")
parser.add_argument('--data_root',type=str,help='Path to data root directory')
parser.add_argument('--features',type=str,help='Path to features relative to data root directory')
parser.add_argument('--output_dir', type=str,default='./checkpoints',help='path to save intermediate checkpoints')

parser.add_argument('--num_classes',type=int,default=48)
parser.add_argument('--sample_rate',type=int,default=1)
parser.add_argument('--num_inputs',type=int,default=128)


# * AGT
parser.add_argument('--model',type=str,default='')
parser.add_argument('--num_queries',type=int,default=10)
parser.add_argument('--num_pos_embed_dict',type=int,default=256)
parser.add_argument('--dim_latent',type=int,default=128)
parser.add_argument('--hidden_dim',type=int,default=256, help='default: 256')
parser.add_argument('--position_embedding',type=str,default='learned', help='fixed, learned, nerf, poseFormer')
parser.add_argument('--dropout',type=float,default=0.1,help='transformer droput')
parser.add_argument('--nheads',type=int,default=8)
parser.add_argument('--dim_feedforward',type=int,default=2048)
parser.add_argument('--enc_layers',type=int,default=1)
parser.add_argument('--dec_layers',type=int,default=3)
parser.add_argument('--pre_norm',action='store_true')
parser.add_argument('--aux_loss',action='store_true')
parser.add_argument('--cuda',action='store_true',help='gpu mode')
parser.add_argument('--eval',action='store_true',help='evaluation mode')
parser.add_argument('--norm_type',type=str,choices=['gn','bn'],default='bn',help="normalization type")
parser.add_argument('--activation',type=str,default='leaky_relu',help="transformer activation type")

# * Matcher
parser.add_argument('--set_cost_class', default=5, type=float,
                        help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_segment', default=5, type=float,
                    help="L1 segment coefficient in the matching cost")
parser.add_argument('--set_cost_siou', default=3, type=float,
                    help="L1 segment coefficient in the matching cost")
# * Loss Coefficients
parser.add_argument('--segment_loss_coef', default=5, type=float)
parser.add_argument('--siou_loss_coef', default=3, type=float)
parser.add_argument('--eos_coef', default=0.1, type=float, help="Relative classification weight of the no-object class")
parser.add_argument('--classification_loss_type', default='focal', type=str, help="classification loss type: focal / ce")
parser.add_argument('--focal_alpha', default=0.25, type=float)
parser.add_argument('--focal_gamma', default=2, type=float)

# * Training
parser.add_argument('--resume',type=str,default='',help='resume from a checkpoint: '
                                                        '/mnt/ssd1/jack/Programs/gtad/packages/activitygraph_transformer/output/checkpoints_/checkpoints_numqueries100_lr1e-5_lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers4_hdim256_sr1_batchsize2_nposembdict512_numinputs100/checkpoint.pth; '
                                                        'output/checkpoints_/checkpoints_numqueries100_lr1e-5_lrdrop1500_dropout0_clipmaxnorm0_weightdecay0_posemblearned_lrjoiner1e-5_nheads4_nenclayers4_ndeclayers4_hdim256_sr1_batchsize2_nposembdict512_numinputs100/checkpoint.pth')
parser.add_argument('--save_checkpoint_every',type=int,default=1000,help='checkpoint saving frequency')
parser.add_argument('--num_workers',type=int,default=0,help='number of workers')
parser.add_argument('--batch_size',type=int,default=2,help='batch_size')
parser.add_argument('--epochs',type=int,default=10,help='number of epochs')
parser.add_argument('--step_size',type=int,default=64,help='number of steps before backpropagation')
parser.add_argument('--start_epoch',type=int,default=0,help='starting epoch')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--lr_joiner', default=0, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--lr_drop', default=100, type=int)
parser.add_argument('--clip_max_norm', default=1, type=float,help='gradient clipping max norm')
parser.add_argument('--eval_every',type=int,default=5,help='eval every ? epochs)')

# * Distributed Training
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--local_rank',type=int,help='local rank')
parser.add_argument('--device', default='cuda:0',help='device to use for training / testing: cpu / cuda:0')

# Variants of Deformable DETR
parser.add_argument('--with_box_refine', default=False, action='store_true')
parser.add_argument('--two_stage', default=False, action='store_true')
# * Backbone
parser.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--num_feature_levels', default=1, type=int, help='number of feature levels, default: 4')
# * Transformer
parser.add_argument('--dec_n_points', default=4, type=int)
parser.add_argument('--enc_n_points', default=4, type=int)
# * Segmentation
parser.add_argument('--masks', action='store_true',
                    help="Train segmentation head if the flag is provided")
parser.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
# * Loss coefficients
parser.add_argument('--mask_loss_coef', default=1, type=float)
parser.add_argument('--dice_loss_coef', default=1, type=float)
parser.add_argument('--cls_loss_coef', default=2, type=float)
parser.add_argument('--bbox_loss_coef', default=5, type=float)
parser.add_argument('--giou_loss_coef', default=2, type=float)

parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
parser.add_argument('--lr_backbone', default=2e-5, type=float)

# parser.add_argument('--space_time', action='store_true', help="whether to use space time attention")
# parser.add_argument('--temporal_deformable', action='store_true', help="whether to use temporal deformable attention")
# parser.add_argument('--graph_self_attention', action='store_true', help="whether to use graph self attention")
parser.add_argument('--use_viewdirs', action='store_true', help="whether to use viewdirs for nerf position embedding")
parser.add_argument('--attention_type', default='divided_space_time', help='space time attention type')

parser.add_argument("--i_embed", type=int, default=0,
                    help='set 0 for default positional encoding, -1 for none')
parser.add_argument("--multires", type=int, default=10,
                    help='log2 of max freq for positional encoding (3D location)')
parser.add_argument("--multires_views", type=int, default=4,
                    help='log2 of max freq for positional encoding (2D direction)')
parser.add_argument("--netchunk", type=int, default=1024 * 64,
                    help='number of pts sent through network in parallel, decrease if running out of memory')
parser.add_argument('--variant', type=str, default='', help="temporal_deformable, graph_self_attention, STAR, TimeSformer, PoseFormer")
parser.add_argument('--USE_WANDB', type=str2bool, default=True, help="use WANDB")
parser.add_argument('--use_gamma_scale', type=str2bool, default=False, help="use gamma scale")
parser.add_argument("--level_concat_dim", type=int, default=2,
                    help='concatenation dim for multi-level feature (default: 1)')
parser.add_argument('--use_nms', type=str2bool, default=False, help="use NMS")
parser.add_argument('--use_CB_loss', type=str2bool, default=True, help="use NMS")
parser.add_argument('--beta', type=float, default=0.9999, help='Hyperparameter for Class balanced loss')

args = parser.parse_args()
print(args)
# USE_WANDB = False # True  # False  #

def main(args):
    # bz = args.batch_size
    # lr = args.lr
    # '''
    if args.USE_WANDB:
        wandb.init(project='agt', config=args, settings=wandb.Settings(_disable_stats=True))
        wandb.run.name = '{}_{}'.format('agt', wandb.run.name.split('-')[-1])
    # '''

    if args.cuda:
        if torch.cuda.device_count() >= 1:
            utils.init_distributed_mode(args)
        device = torch.device(args.device) 
    else:
        device = torch.device('cpu')
    # fix the seed for reproducibility
    if args.cuda:
        seed = args.seed + utils.get_rank()
    else:
        seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # set up model
    model, criterion, postprocessors = build_model(args)

    model_without_ddp = model
    if args.cuda:
        # criterion.to(device)
        if args.distributed:
            if args.mp:
                model = torch.nn.parallel.DistributedDataParallel(model)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)

            model_without_ddp = model.module
        # else:
        #     model = torch.nn.DataParallel(model)  # .to(device)
        #     model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    # set up model training
    param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if "joiner" not in n and p.requires_grad]},
        {"params": [p for n, p in model_without_ddp.named_parameters() if "joiner" in n and p.requires_grad], "lr": args.lr_joiner,},]



    # datasets build
    dataset_train = build_dataset(mode="training", args=args)
    dataset_test = build_dataset(mode="testing", args=args)

    if args.cuda and args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=False)
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, 1, sampler=sampler_test, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # output and checkpoints directory
    checkpoint_dir = args.output_dir
    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except OSError:
            pass
 
    if args.resume:
        checkpoint = Path(args.resume)
        assert checkpoint.exists()

        checkpoint = torch.load(args.resume, map_location='cpu')
        try:
            print('Load full model.')
            model_without_ddp.load_state_dict(checkpoint['model'])
            load_model_except_class_embed = False
        except:
            print('Cannot load full model. Load part model.')
            # load part ckpt
            model_dict = model_without_ddp.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict and 'class_embed' not in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            model_without_ddp.load_state_dict(model_dict)
            load_model_except_class_embed = True

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint and not load_model_except_class_embed:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        print('load checkpoint from {} successfully.'.format(args.resume))


    print("Start Training")
    start_time = time.time() 
    optimizer.zero_grad()
    for epoch in range(args.start_epoch, args.epochs):
        if args.cuda and args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(epoch, args.clip_max_norm, model, criterion, data_loader_train, optimizer,
                                      lr_scheduler, device, position_embedding=args.position_embedding)
        # train_stats = {}
        # print('train_stats: ', train_stats)

        if args.output_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_paths = [checkpoint_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_every == 0:
                checkpoint_paths.append(checkpoint_dir / f'checkpoint{epoch:05}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch, 'args': args,}, checkpoint_path) 
        
        # evaluation
        if epoch % args.eval_every == 0:
            test_stats = evaluate(epoch, model, criterion, postprocessors, data_loader_test, args.output_dir, args.dataset, device, position_embedding=args.position_embedding)
            # print('test_stats: ', test_stats)
        else:
            test_stats = {}
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()},'epoch': epoch, 'n_parameters': n_parameters}
        # print('log_stats: ', log_stats)
        # '''
        if args.USE_WANDB:
            wandb.log(log_stats, step=epoch+1)
        # '''
        if args.output_dir and utils.is_main_process():
            with (checkpoint_dir / 'log.json').open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        lr_scheduler.step()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    main(args)



