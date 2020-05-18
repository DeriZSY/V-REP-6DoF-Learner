import sim
import math
import numpy as np
import utils
import os
from data_utils import get_pose_mat
import argparse
from config import make_cfg
import copy
import datetime
import json
import time


class PoseGenerator:
    def __init__(self, cfg):
        self.mode = 'robot' if cfg.robot_name != '' else 'link'
        self.num_steps = 0
        self.cur_step = 0
        self.step_size = 0.01

        self.num_euler_step = cfg.pose_generator.num_euler_step
        self.num_trans_step = cfg.pose_generator.num_euler_step

        self.trans_bounds = [
            cfg.pose_generator.trans_x,
            cfg.pose_generator.trans_y,
            cfg.pose_generator.trans_z,
        ]

        self.rot_bounds = [
            cfg.pose_generator.euler_i,
            cfg.pose_generator.euler_j,
            cfg.pose_generator.euler_k,
        ]

        self.trans_steps = []
        self.rot_steps = []
        self.comb_steps = []

        self.position_init = None
        self.rot_init = None

        if cfg.pose_generator.use_marker:
            marker_path = os.path.join(cfg.data_root, 'marker_pose.json')
            self.init_marker(marker_path)
        else:
            self.init()

    def set_pose_init(self, pose_init):
        position_init, rot_init = pose_init
        self.set_pos_init(position_init)
        self.set_rot_init(rot_init)

    def set_pos_init(self, position_init):
        assert len(position_init) == 3, f"Error dimension for pose init:{position_init}"
        self.position_init = position_init

    def set_rot_init(self, rot_init):
        assert len(rot_init) == 3 or len(rot_init) == 4, f"Error dimension for rot init:{rot_init}"
        self.rot_init = rot_init

    @staticmethod
    def get_bounded_grid(bound_x, bound_y, bound_z, step_count):
        meshgrids = []
        for x_iter in np.linspace(bound_x[0], bound_x[1], step_count):
            for y_iter in np.linspace(bound_y[0], bound_y[1], step_count):
                for z_iter in np.linspace(bound_z[0], bound_z[1], step_count):
                    meshgrids.append([x_iter, y_iter, z_iter])

        return meshgrids

    def init_marker(self, marker_path):
        marker_file = json.load(open(marker_path, 'r'))
        for marker_id in marker_file.keys():
            pose = marker_file[marker_id]
            self.comb_steps.append(pose)
        self.num_steps = len(self.comb_steps)

    def init(self):
        if self.mode == 'robot':

            trans_steps = self.get_bounded_grid([0, self.num_trans_step-1], [0, self.num_trans_step-1],
                                                [0, self.num_trans_step-1], self.num_trans_step)
            self.trans_steps = np.array(trans_steps) * self.step_size

            rot_steps = [
                [-0.6698113083839417, -0.7309706807136536, 0.09623405337333679, -0.08818229287862778],
                [-0.6443228125572205, -0.7365765571594238, -0.032159700989723206, -0.20315414667129517],
                [-0.6749482750892639, -0.719801664352417, 0.15957611799240112, -0.029468879103660583],
                [-0.6718955039978027, -0.7288868427276611, -0.08040499687194824, 0.10401153564453125],
                [-0.658875584602356, -0.7205038666725159, 0.15424583852291107, -0.15155519545078278],
                [-0.4604227840900421, -0.883388876914978, 0.07919453829526901, -0.036976978182792664],
                [-0.8629612922668457, -0.49765756726264954, 0.04692428559064865, -0.07373727858066559],
                [-0.8579992651939392, -0.5029014348983765, -0.10364025831222534, 0.013800282031297684],
                [-0.8555877208709717, -0.48933735489845276, 0.12195773422718048, -0.11683043092489243],
                [-0.4576055705547333, -0.8807231187820435, 0.032160647213459015, -0.11787539720535278],
                [-0.44609710574150085, -0.8754066228866577, -0.10693252831697464, 0.152418851852417],
                [-0.6936686635017395, -0.6812778115272522, -0.23349764943122864, -0.013006586581468582],
                [-0.7277641296386719, -0.66231369972229, -0.07782546430826187, -0.16015379130840302],
                [-0.6719654202461243, -0.7216885089874268, 0.1592566967010498, -0.04766680672764778],
            ]
            rot_steps = np.array(rot_steps)

            self.rot_steps = (rot_steps - rot_steps[0])
            self.num_steps = len(self.trans_steps) * len(rot_steps)
        elif self.mode == 'link':
            assert len(self.trans_bounds) == 3, \
                f"Error length for translation bounds, expecting 3 but get:{len(self.trans_bounds)}"
            trans_x, trans_y, trans_z = self.trans_bounds
            trans_steps = self.get_bounded_grid(trans_x, trans_y, trans_z, self.num_trans_step)
            self.trans_steps = np.array(trans_steps)

            assert len(self.rot_bounds) == 3, \
                f"Error length for rotation bounds, expecting 3 but get:{len(self.rot_bounds)}"
            rot_i, rot_j, rot_k = self.rot_bounds
            rot_steps = self.get_bounded_grid(rot_i, rot_j, rot_k, self.num_euler_step)
            # FIXME: dirt fix for invalid rotations
            weak_remove_ids = [5, 8, 9]
            # remove_ids = [6, 18, 19, 20, 21, 22, 25, 26]
            remove_ids = []
            rot_steps_clean = []
            for id, rot_step in enumerate(rot_steps):
                if id not in remove_ids:
                    rot_steps_clean.append(rot_step)
            rot_steps = rot_steps_clean

            self.rot_steps = np.array(rot_steps)
            self.num_steps = len(self.trans_steps) * len(self.rot_steps)
        else:
            raise NotImplementedError(f"Required type:{self.mode} not implemented at the moment")

    def __iter__(self):
        return

    def num_steps(self):
        return self.num_steps()

    def _step_mapping(self):
        step_trans = math.floor(self.cur_step / len(self.rot_steps))
        setp_rot = math.floor(self.cur_step %  len(self.rot_steps))
        return step_trans, setp_rot

    def curr_step(self):
        return self.cur_step

    def __next__(self):
        if self.cur_step < self.num_steps:
            if len(self.comb_steps) > 0:
                trans, rot = self.comb_steps[self.cur_step]
            else:
                trans_id, rot_id = self._step_mapping()
                trans = self.trans_steps[trans_id]
                rot = self.rot_steps[rot_id]
            self.cur_step += 1

            # print(f"Processing {self.cur_step}/{self.num_steps}")
            if self.position_init is not None and self.rot_init is not None:
                return trans + self.position_init, rot + self.rot_init
            else:
                return trans, rot
        else:
            raise StopIteration


class VrepRobot(object):
    """ Warper for v-rep robot control"""
    def __init__(self, cfg, pose_generator):
        sim.simxFinish(-1)  # Just in case, close all opened connections
        self.robot_dicts = {}
        self.camera_dicts = {}
        self.dump_rt = cfg.dump_rt
        self.frame_id = 0

        self.pose_generator = pose_generator
        self.use_additive_pose = cfg.pose_generator.use_additive
        self.robot_name = cfg.robot_name if cfg.robot_name != '' else None
        self.target_name = cfg.target_name if cfg.target_name != '' else None
        self.default_cam = cfg.default_cam if cfg.default_cam != '' else None
        self.cam_names = cfg.cam_names

        self.clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
        assert self.clientID != -1, "Failed to connect to simulation server."

        self.robot_handle = None
        self.obj_handle = None
        self.target_handle = None
        self.frame_info_list = []
        self.meta_data = None
        self.marker_poses = dict()

        # Set path for data restoration
        self.data_root = cfg.data_root
        assert os.path.isdir(os.path.dirname(cfg.data_root))
        if not os.path.isdir(cfg.data_root):
            os.makedirs(cfg.data_root)

        self.marker_dir = os.path.join(self.data_root, 'marker_pose.json')
        self.im_path = os.path.join(self.data_root, 'rgb', '{}.png')
        self.depth_path = os.path.join(self.data_root, 'depth', '{}')
        self.frame_info_path = os.path.join(self.data_root, 'frame', '{}.json')
        self.meta_path = os.path.join(self.data_root, 'meta.json')

    def get_camera_data(self, cam_info, need_depth=False):

        assert isinstance(cam_info, dict), "Error, camera info dict not found, cannot obtain image..."

        # Get color image from simulation
        cam_handle = cam_info['handle']

        sim_ret, resolution, raw_image = sim.simxGetVisionSensorImage(self.clientID, cam_handle, 0,
                                                                      sim.simx_opmode_blocking)

        while len(resolution) <= 1:
            sim_ret, resolution, raw_image = sim.simxGetVisionSensorImage(self.clientID, cam_handle, 0,
                                                                          sim.simx_opmode_blocking)

        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float) / 255
        color_img[color_img < 0] += 1
        color_img *= 255

        color_img = np.fliplr(color_img)
        # color_img = np.flipud(color_img)
        color_img = color_img.astype(np.uint8)

        if need_depth:
            # Get depth image from simulation
            sim_ret, resolution, depth_buffer = sim.simxGetVisionSensorDepthBuffer(self.clientID, cam_handle,
                                                                                   sim.simx_opmode_blocking)
            depth_img = np.asarray(depth_buffer)
            depth_img.shape = (resolution[1], resolution[0])
            depth_img = np.fliplr(depth_img)
            # depth_img = np.flipud(depth_img)
            zNear = 0.01
            zFar = 10
            depth_img = depth_img * (zFar - zNear) + zNear

            depth_scale = cam_info['depth_scale']
            depth_img = depth_img * depth_scale
        else:
            depth_img = np.array([])

        return color_img, depth_img, cam_info['name']

    def setup_all(self):
        self.setup_robot()
        self.setup_sim_cameras()

    def setup_robot(self):
        """ Setup robot (if available) """
        if self.robot_name is not None:
            sim_ret, self.robot_handle = sim.simxGetObjectHandle(self.clientID, self.robot_name, sim.simx_opmode_blocking)
            sim_ret, self.target_handle = sim.simxGetObjectHandle(self.clientID, self.target_name, sim.simx_opmode_blocking)
        else:
            # set robot handle to target object if no robot used
            sim_ret, self.target_handle = sim.simxGetObjectHandle(self.clientID, self.target_name, sim.simx_opmode_blocking)
            self.robot_handle = self.target_handle

    def setup_sim_cameras(self):
        """ Setup infromation form vision_sensors """

        def _get_K(resolution):
            width, height = resolution
            view_angle = (54.70 / 180) * math.pi
            fx = (width / 2.) / math.tan(view_angle / 2)
            fy = fx
            cx = width / 2.
            cy = height / 2.
            # cam_intrinsics = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            cam_intrinsics_andyzeng = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
            return cam_intrinsics_andyzeng

        assert len(self.cam_names) != 0, "No camera to add, exiting ..."

        for cam_name in self.cam_names:
            sim_ret, cam_handle = sim.simxGetObjectHandle(self.clientID, cam_name,
                                                          sim.simx_opmode_blocking)

            _, resolution, _ = sim.simxGetVisionSensorImage(self.clientID, cam_handle, 0,
                                                                          sim.simx_opmode_blocking)
            cam_intrinsic = _get_K(resolution)

            # Get camera pose and intrinsics in simulation
            sim_ret, cam_position = sim.simxGetObjectPosition(self.clientID, cam_handle, -1,
                                                              sim.simx_opmode_blocking)
            sim_ret, cam_quat = sim.simxGetObjectQuaternion(self.clientID, cam_handle, -1,
                                                                    sim.simx_opmode_blocking)
            cam_pose = get_pose_mat((cam_position, cam_quat))

            cam_depth_scale = 1
            cam_info_dict = {
                'name': cam_name,
                'handle': cam_handle,
                'pose': cam_pose.tolist(),
                'intrinsics': cam_intrinsic.tolist(),
                'depth_scale': cam_depth_scale,
                'im_shape': [resolution[1], resolution[0]]
            }
            self.camera_dicts[cam_name] = cam_info_dict

    def _get_pose(self, obj_handle, use_quat=True):
        assert obj_handle is not None, "Object handler isn't set"
        sim_ret, position = sim.simxGetObjectPosition(self.clientID, obj_handle, -1, sim.simx_opmode_blocking)
        if use_quat:
            sim_ret, orientation = sim.simxGetObjectQuaternion(self.clientID, obj_handle, -1, sim.simx_opmode_blocking)
        else:
            sim_ret, orientation = sim.simxGetObjectOrientation(self.clientID, obj_handle, -1, sim.simx_opmode_blocking)

        pose = (np.array(position), np.array(orientation))
        return pose

    def _set_pose(self, obj_handle, target_pose):
        assert obj_handle is not None, "Object handler isn't set"
        if len(target_pose) != 2 and not isinstance(target_pose, tuple):
            raise NotImplementedError("Only original V-REP format is allowed for robot control at present")

        target_pos, target_rot = target_pose
        sim.simxSetObjectPosition(self.clientID, obj_handle, -1, target_pos, sim.simx_opmode_blocking)
        if len(target_rot) == 4:
            sim.simxSetObjectQuaternion(self.clientID, obj_handle, -1, target_rot, sim.simx_opmode_blocking)
        elif len(target_rot) == 3:
            sim.simxSetObjectOrientation(self.clientID, obj_handle, -1, target_rot, sim.simx_opmode_blocking)
        import time
        time.sleep(0.1)

    def _get_meta(self):
        if self.meta_data is not None:
            return

        cams_info = copy.deepcopy(self.camera_dicts)
        for cam in cams_info.keys():
            cams_info[cam].pop('handle')

        timestamp = datetime.datetime.now().timestamp()

        meta_data = {
            'cam_default': self.default_cam,
            'cam_info': cams_info,
            'robot_info': self.robot_handle,
            'time': timestamp
        }
        self.meta_data = meta_data

    def _collect_data(self, cam_name):
        cam_info = self.camera_dicts[cam_name]
        rgb_img, depth_img, cam_name = self.get_camera_data(cam_info)
        Tow = get_pose_mat(self._get_pose(self.target_handle))
        rgb_path = self.im_path.format(self.frame_id)
        depth_path = self.depth_path.format(self.frame_id)
        info_path = self.frame_info_path.format(self.frame_id)

        frame_data_dict = {
            'frame_id': self.frame_id,
            'cam_name': cam_name,
            'pose': Tow.tolist(),
            'im_rgb': rgb_img,
            'im_depth': depth_img,
            'rgb': rgb_path,
            'depth': depth_path,
            'info_path': info_path
        }

        self.frame_info_list.append(frame_data_dict)
        self.frame_id += 1
        if self.dump_rt:
            self.frame_info_list = [self.frame_info_list[-1]]
            self.dump_data()

    def collect_data(self):
        if self.default_cam is not None:
            assert isinstance(self.default_cam, str), \
                f"Data type for default camera{self.default_cam} not understood"
            self._collect_data(self.default_cam)
        else:
            for cam_name in self.camera_dicts.keys():
                self._collect_data(cam_name)

    def dump_data(self, save_depth=False):
        import json

        def _check_save_dir():
            path_list = [self.im_path, self.im_path, self.depth_path, self.frame_info_path]
            for path in path_list:
                folder_dir = os.path.dirname(path)
                if not os.path.isdir(folder_dir):
                    os.makedirs(folder_dir)

        _check_save_dir()

        if not os.path.isfile(self.meta_path):
            self._get_meta()
            with open(self.meta_path, 'w') as f:
                json.dump(self.meta_data, f, indent=4)

        for frame_info in self.frame_info_list:
            # Save rgb image
            im_rgb = frame_info['im_rgb']
            rgb_path = frame_info['rgb']
            frame_info.pop('im_rgb')
            utils.save_rgb(im_rgb, rgb_path)

            if save_depth:
                # Save depth
                im_depth = frame_info['im_depth']
                depth_path = frame_info['depth']
                utils.save_depth(im_depth, depth_path)
            frame_info.pop('im_depth')

            # Save frame info
            info_path = frame_info['info_path']
            with open(info_path, 'w') as f:
                json.dump(frame_info, f, indent=4)

    def close(self):
        sim.simxGetPingTime(self.clientID)
        sim.simxFinish(self.clientID)
        if not self.dump_rt:
            self.dump_data()

    def run(self):
        # get current pose of target
        if self.use_additive_pose:
            self.pose_generator.set_pose_init(self._get_pose(self.robot_handle, use_quat=True))

        from tqdm import tqdm
        for num_iter in tqdm(range(self.pose_generator.num_steps)):
            next_pose = self.pose_generator.__next__()
            trans_step, rot_step = next_pose
            self._set_pose(self.robot_handle, (trans_step, rot_step))
            self.collect_data()
        self.close()

    def run_manual(self):
        from tqdm import tqdm
        for num_iter in tqdm(range(self.pose_generator.num_steps())):
            next_pose = self.pose_generator.__next__()
            trans_step, rot_step = next_pose
            self._set_pose(self.robot_handle, (trans_step, rot_step))
            self.collect_data()
        self.close()

    def move_and_collect(self, target_obj_name):
        def get_pose():
            sim_ret, target_handle = sim.simxGetObjectHandle(self.clientID, target_obj_name, sim.simx_opmode_blocking)
            sim_ret, orientation = sim.simxGetObjectOrientation(self.clientID, target_handle, -1,
                                                                sim.simx_opmode_blocking)
            sim_ret, position = sim.simxGetObjectPosition(self.clientID, target_handle, -1, sim.simx_opmode_blocking)
            return position, orientation

        def set_pose(pos, rot):
            sim_ret, target_handle = sim.simxGetObjectHandle(self.clientID, target_obj_name, sim.simx_opmode_blocking)
            sim.simxSetObjectPosition(self.clientID, target_handle, -1, pos, sim.simx_opmode_blocking)
            sim.simxSetObjectOrientation(self.clientID, target_handle, -1, rot, sim.simx_opmode_blocking)

        t_mode = 1 # 1:xy 2:yz 3: xz
        r_mode = 1 # 1: x, 2:y
        t_step = 0.05
        r_step = 5/180*3.14

        def mod_trans(pos, cmd):
            idx1 = 0
            idx2 = 1
            if t_mode == 1:
                idx1 = 0
                idx2 = 1
            elif t_mode == 2:
                idx1 = 2
                idx2 = 1
            elif t_mode == 3:
                idx1 = 2
                idx2 = 0

            if cmd == 'w':
                pos[idx1] += t_step
            elif cmd == 's':
                pos[idx1] -= t_step
            elif cmd == 'a':
                pos[idx2] -= t_step
            elif cmd == 'd':
                pos[idx2] += t_step
            return pos

        def mod_rot(rot, cmd):
            idx = 0
            if r_mode == 1:
                idx = 0
            elif r_mode == 2:
                idx = 1

            if cmd == 'j':
                rot[idx] += r_step
            elif cmd == 'k':
                rot[idx] -= r_step
            return rot

        while True:
            pos, rot = get_pose()
            cmd = input()
            if cmd == 'm':
                t_mode += 1
                if t_mode == 4:
                    t_mode = 1
            elif cmd == 'n':
                r_mode += 1
                if r_mode == 3:
                    r_mode = 1
            elif cmd in ['w', 'a', 's', 'd']:
                pos = mod_trans(pos, cmd)
            elif cmd == 'q':
                break
            else:
                rot = mod_rot(rot, cmd)
            set_pose(pos, rot)
            self.check_pose(target_obj_name, mode='marker')

    def check_pose(self, target_obj_name, mode='output'):
        # if self.obj_handle is None:
        sim_ret, target_handle = sim.simxGetObjectHandle(self.clientID, target_obj_name, sim.simx_opmode_blocking)
        sim_ret, orientation = sim.simxGetObjectOrientation(self.clientID, target_handle, -1, sim.simx_opmode_blocking)
        sim_ret, position = sim.simxGetObjectPosition(self.clientID, target_handle, -1, sim.simx_opmode_blocking)
        sim_ret, quat = sim.simxGetObjectQuaternion(self.clientID, target_handle, -1, sim.simx_opmode_blocking)
        if mode == 'output':
            print(f"trans:{position}\nrot:{quat}\n")
        elif mode == 'marker':
            import json
            self.marker_poses = json.load(open(self.marker_dir, 'r'))
            marker_id = len(list(self.marker_poses.keys()))
            self.marker_poses[marker_id] = (position, quat)
            json.dump(self.marker_poses, open(self.marker_dir, 'w'), indent=4)
            # print(f"Size:{marker_id}\ntrans:{position}\nrot:{orientation}\n\n")
        else:
            raise NotImplementedError(f"Method not implement for mode:{mode}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default="configs/UR.yaml", type=str)
    parser.add_argument("--data_root", default="", type=str)
    parser.add_argument("--dump_rt", default=True, type=bool)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = make_cfg(args)

    pose_generator = PoseGenerator(cfg)
    robot_controller = VrepRobot(cfg, pose_generator)
    robot_controller.setup_all()
    robot_controller.run()

    # print("ready")
    # key = input()
    # id = 0
    # while(True):
    #     id += 1
        # time.sleep(0.3)
        # print(f"saving pose:{id}")
        # ur_sim.check_pose('Franka_target', mode='marker')
        # ur_sim.move_and_collect('Franka_target')
