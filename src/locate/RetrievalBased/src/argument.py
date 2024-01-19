import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Smth-Else')

    # Path related arguments

    parser.add_argument('--model',
                        default='coord')
    parser.add_argument('--root_frames', type=str, default='', help='path to the folder with frames')
    parser.add_argument('--json_data_train', type=str, help='path to the json file with train video meta data')
    parser.add_argument('--json_data_val', type=str, help='path to the json file with validation video meta data')
    parser.add_argument('--json_file_labels', type=str, help='path to the json file with ground truth labels')
    parser.add_argument('--img_feature_dim', default=256, type=int, metavar='N',
                        help='intermediate feature dimension for image-based features')
    parser.add_argument('--coord_feature_dim', default=128, type=int, metavar='N',
                        help='intermediate feature dimension for coord-based features')
    parser.add_argument('--clip_gradient', '-cg', default=5, type=float,
                        metavar='W', help='gradient norm clipping (default: 5)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--size', default=224, type=int, metavar='N',
                        help='primary image input size')
    parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', '-b', default=72, type=int,
                        metavar='N', help='mini-batch size (default: 72)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_steps', default=[24, 35, 45], type=float, nargs="+",
                        metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=0.0001, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print_freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--log_freq', '-l', default=10, type=int,
                        metavar='N', help='frequency to write in tensorboard (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--num_classes', default=192, type=int,
                        help='num of class in the model')
    parser.add_argument('--num_boxes', default=30, type=int,
                        help='num of boxes for each image')
    parser.add_argument('--num_frames', default=4, type=int,
                        help='num of frames for the model')
    parser.add_argument('--action_feature_dim', default=8, type=int,
                        help='action feature dim')
    parser.add_argument('--dataset', default='crosstask',
                        help='which dataset to train')
    parser.add_argument('--logdir', default='./logs',
                        help='folder to output tensorboard logs')
    parser.add_argument('--logname', default='exp',
                        help='name of the experiment for checkpoints and logs')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--fine_tune', help='path with ckpt to restore')
    parser.add_argument('--tracked_boxes', type=str, help='choose tracked boxes')
    parser.add_argument('--shot', default=5)
    parser.add_argument('--restore_i3d')
    parser.add_argument('--restore_custom')

    parser.add_argument('--lang_model', default='', type=str, metavar='LANG',
                        help='language model (default: generative)')
    parser.add_argument('--dataset_mode', default='proc_plan', type=str, metavar='DATA',
                        help='dataset mode (default: '')')
    parser.add_argument('--model_type', default='model_T', type=str, metavar='MODELT',
                        help='forward dynamics model (model_T) or conjugate dynamics model (model_P)')
    parser.add_argument('--max_sentence_len', default=3, type=int, metavar='MAXMESS',
                        help='max message length (default: 5)')
    parser.add_argument('--max_traj_len', default=3, type=int, metavar='MAXTRAJ',
                        help='max trajectory length (default: 54)')
    parser.add_argument('--roi_feature', type=str2bool,
                            default=True,
                            help='Using RoIAlign Feature')
    parser.add_argument('--random_coord', type=str2bool,
                            default=False,
                            help='Use random coord')
    parser.add_argument('--use_rnn', type=str2bool,
                            default=False,
                            help='Use RNN')
    return parser.parse_args()