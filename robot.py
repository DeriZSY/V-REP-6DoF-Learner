import sim
import math
import numpy as np
import utils
import os
from data_utils import get_pose_mat
import argparse
from config import make_cfg


class PoseGenerator:
    def __init__(self, cfg):
        self.mode = 'robot' if cfg.robot_name != '' else 'link'
        self.num_steps = 0
        self.cur_step = 0
        self.step_count = 4

        self.trans_steps = []
        self.rot_steps = []

        self.init()

    def init(self):
        if self.mode == 'robot':
            space_mat = []
            for z_iter in range(self.step_count):
                for y_iter in range(self.step_count):
                    for x_iter in range(self.step_count):
                        space_mat.append([x_iter, y_iter, z_iter])

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
            self.trans_steps = space_mat
            self.rot_steps = (rot_steps - rot_steps[0])
            self.num_steps = self.step_count ** 3 * len(rot_steps)
        elif self.mode == 'link':
            pass
        else:
            raise NotImplementedError(f"Required type:{self.mode} not implemented at the moment")

    def __iter__(self):
        return self

    def _step_mapping(self):
        step_trans = math.floor(self.cur_step / len(self.rot_steps))
        setp_rot = math.floor(self.cur_step % self.step_count)
        return step_trans, setp_rot

    def curr_step(self):
        return self.cur_step

    def __next__(self):
        if self.cur_step < self.num_steps:
            trans_id, rot_id = self._step_mapping()
            trans = self.trans_steps[trans_id]
            rot = self.rot_steps[rot_id]
            self.cur_step += 1
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

        # Set path for data restoration
        self.data_root = cfg.data_root
        assert os.path.isdir(os.path.dirname(cfg.data_root))
        if not os.path.isdir(cfg.data_root):
            os.makedirs(cfg.data_root)

        self.im_path = os.path.join(self.data_root, 'rgb', '{}.png')
        self.depth_path = os.path.join(self.data_root, 'depth', '{}')
        self.frame_info_path = os.path.join(self.data_root, 'frame', '{}.json')
        self.meta_path = os.path.join(self.data_root, 'meta.json')

    def get_camera_data(self, cam_info):
        # if cam_info is None:
        #     cam_info = self.camera_dicts[self.default_cam]
        # elif cam_info is not None and isinstance(cam_info, str):
        #     cam_info = self.camera_dicts[cam_info]

        assert isinstance(cam_info, dict), "Error, camera info dict not found, cannot obtain image..."

        # Get color image from simulation
        cam_handle = cam_info['handle']
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

        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = sim.simxGetVisionSensorDepthBuffer(self.clientID, cam_handle,
                                                                               sim.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        # FIXME: whether need to flip?
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        # depth_img = np.flipud(depth_img)
        zNear = 0.01
        zFar = 10
        depth_img = depth_img * (zFar - zNear) + zNear

        depth_scale = cam_info['depth_scale']
        depth_img = depth_img * depth_scale
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

    def _get_pose(self, obj_handle):
        assert obj_handle is not None, "Object handler isn't set"
        sim_ret, position = sim.simxGetObjectPosition(self.clientID, obj_handle, -1, sim.simx_opmode_blocking)
        sim_ret, orientation = sim.simxGetObjectQuaternion(self.clientID, obj_handle, -1, sim.simx_opmode_blocking)
        pose = (position, orientation)
        return pose

    def _set_pose(self, obj_handle, target_pose):
        assert obj_handle is not None, "Object handler isn't set"
        if len(target_pose) != 2 and not isinstance(target_pose, tuple):
            raise NotImplementedError("Only original V-REP format is allowed for robot control at present")

        target_pos, target_rot = target_pose
        sim.simxSetObjectPosition(self.clientID, obj_handle, -1, target_pos, sim.simx_opmode_blocking)
        sim.simxSetObjectQuaternion(self.clientID, obj_handle, -1, target_rot, sim.simx_opmode_blocking)

    def _get_meta(self):
        if self.meta_data is not None:
            return

        import copy
        import datetime

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
        position, orientation = self._get_pose(self.robot_handle)
        rot_init = orientation
        for next_pose in self.pose_generator:
            trans_step, rot_step = next_pose
            self._set_pose(self.robot_handle, (position+trans_step, rot_init+rot_step))
            self.collect_data()
        self.close()

    def get_test_pose(self, target_obj_name):
        if self.obj_handle is None:
            sim_ret, self.robot_handle = sim.simxGetObjectHandle(self.clientID, target_obj_name, sim.simx_opmode_blocking)
        sim_ret, orientation = sim.simxGetObjectQuaternion(self.clientID, self.obj_handle, -1, sim.simx_opmode_blocking)
        return orientation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default="configs/lnikOnly.yaml", type=str)
    parser.add_argument("--data_root", default="data/vrep_link", type=str)
    parser.add_argument("--dump_rt", default=True, type=bool)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = make_cfg(args)

    pose_generator = PoseGenerator(cfg)
    ur_sim = VrepRobot(cfg, pose_generator)
    ur_sim.setup_all()
    ur_sim.run()

    # while(True):
    #     key = input()
    #     print("{}".format(ur_sim.get_test_pose('link6')))
