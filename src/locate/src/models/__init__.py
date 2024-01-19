from .agt import build
from .deformable_detr import build as deformable_build


def build_model(args):
    # if args.with_box_refine or args.two_stage:
    return deformable_build(args)
    # else:
    #     return build(args)
