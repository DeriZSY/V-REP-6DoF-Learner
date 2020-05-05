from .yacs import CfgNode as CN
import argparse
import os

cfg = CN()

# ======== meta_param
cfg.data_root = 'data/vrep'  # directory to save acquired data
cfg.dump_rt = True  # whether to dump data on-the-fly
cfg.robot_name = ''
cfg.target_name = ''
cfg.cam_names = []
cfg.default_cam = ''

# ======= pose_generator
cfg.pose_generator = CN()
cfg.pose_generator.use_additive = False
cfg.pose_generator.trans_x = [] # [min_x, max_x]
cfg.pose_generator.trans_y = [] # [min_y, max_y]
cfg.pose_generator.trans_z = [] # [min_z, max_z]
cfg.pose_generator.num_trans_step = 4 # Num steps to interpolate between min max
cfg.pose_generator.euler_i = [] # [min, max]
cfg.pose_generator.euler_j = []
cfg.pose_generator.euler_k = []
cfg.pose_generator.num_euler_step = 4


def parse_cfg(cfg, args):
    if os.path.isdir(os.path.dirname(args.data_root)):
        cfg.data_root = args.data_root

    cfg.dump_rt = args.dump_rt


def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)

    assert cfg.pose_generator.num_trans_step > 0, \
        f"Found trans steps:{cfg.pose_generator.num_trans_step} which should be >= 0"

    assert cfg.pose_generator.num_euler_step > 0, \
        f"Found euler steps:{cfg.pose_generator.num_euler_step} which should be >= 0"

    return cfg
