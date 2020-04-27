from .yacs import CfgNode as CN
import argparse
import os

cfg = CN()

# ======== meta_param
cfg.data_path = 'data/vrep'  # directory to save acquired data
cfg.dump_rt = True  # whether to dump data on-the-fly
cfg.robot_name = ''
cfg.target_name = ''
cfg.cam_names = []
cfg.default_cam = ''


def parse_cfg(cfg, args):
    cfg.data_root = args.data_root
    cfg.dump_rt = args.dump_rt


def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg, args)
    if cfg.robot_name == '':
        cfg.robot_name = None

    return cfg
