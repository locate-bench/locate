import torch.utils.data
import torchvision

def build_dataset(mode, args):
    if args.dataset == 'thumos':
        from datasets.thumos import build_thumos_detection
        return build_thumos_detection(mode=mode,args=args)
    elif args.dataset == 'babel':
        from datasets.babel import build_babel_detection
        return build_babel_detection(mode=mode,args=args)
    raise ValueError(f'dataset {args.dataset} not supported')
