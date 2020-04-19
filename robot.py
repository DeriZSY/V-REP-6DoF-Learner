import sim
import math
import numpy as np
import utils
import os
from data_utils import get_pose_mat


class VrepRobot:
    """ Warper for v-rep robot control"""
    def __init__(self, data_root, dump_rt=True):
        sim.simxFinish(-1)  # Just in case, close all opened connections
        self.robot_dicts = {}
        self.camera_dicts = {}
        self.dump_rt = dump_rt
        self.frame_id = 0

        self.robot_name = 'UR10_target'
        self.cam_names = ['Vision_sensor', 'Vision_sensor0', 'Vision_sensor1', 'Vision_sensor2']
        self.default_cam = 'Vision_sensor'

        self.clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
        assert self.clientID != -1, "Failed to connect to simulation server."

        self.obj_handle = None

        assert os.path.isdir(os.path.dirname(data_root))
        if not os.path.isdir(data_root):
            os.makedirs(data_root)

        self.data_root = data_root
        self.im_path = os.path.join(self.data_root, 'rgb', '{}.png')
        self.depth_path = os.path.join(self.data_root, 'depth', '{}')
        self.frame_info_path = os.path.join(self.data_root, 'frame', '{}.json')
        self.meta_path = os.path.join(self.data_root, 'meta.json')

        self.frame_info_list = []
        self.meta_data = None

    def get_camera_data(self, cam_info=None):
        if cam_info is None:
            cam_info = self.camera_dicts[self.default_cam]
        elif cam_info is not None and isinstance(cam_info, str):
            cam_info = self.camera_dicts[cam_info]

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

    def setup_robot(self, robot_name=None):
        if robot_name is not None:
            self.robot_name = robot_name
        sim_ret, self.robot_handle = sim.simxGetObjectHandle(self.clientID, self.robot_name, sim.simx_opmode_blocking)

    def setup_sim_cameras(self, cam_names=None):
        def _get_K(resolution):
            width, height = resolution
            view_angle = (54.70 / 180) * math.pi
            fx = (width / 2.) / math.tan(view_angle / 2)
            fy = fx
            cx = width / 2.
            cy = height / 2.

            cam_intrinsics_andyzeng = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
            cam_intrinsics = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            return cam_intrinsics
        if cam_names is None:
            cam_names = self.cam_names

        if cam_names is None or len(cam_names) == 0:
            print("No camera to add, exiting ...")

        for cam_name in cam_names:
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
            cam_pose = get_pose_mat(cam_position, cam_quat)

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

    def get_robot_pose(self):
        assert self.robot_handle is not None, "Robot handler isn't set"
        sim_ret, position = sim.simxGetObjectPosition(self.clientID, self.robot_handle, -1, sim.simx_opmode_blocking)
        sim_ret, orientation = sim.simxGetObjectQuaternion(self.clientID, self.robot_handle, -1, sim.simx_opmode_blocking)
        pose = (position, orientation)
        return pose

    def set_pose(self, target_pose):
        assert self.robot_handle is not None, "Robot handler isn't set"
        if len(target_pose) != 2 and not isinstance(target_pose, tuple):
            raise NotImplementedError("Only original V-REP format is allowed for robot control at present")

        target_pos, target_rot = target_pose
        sim.simxSetObjectPosition(self.clientID, self.robot_handle, -1, target_pos, sim.simx_opmode_blocking)
        sim.simxSetObjectQuaternion(self.clientID, self.robot_handle, -1, target_rot, sim.simx_opmode_blocking)

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

    def collect_data(self, mutli_collect=True):

        if not mutli_collect:
            rgb_img, depth_img, cam_name = self.get_camera_data()
            Tow = self.get_obj_pose()

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
        else:
            for cam_name in self.camera_dicts.keys():
                cam_info = self.camera_dicts[cam_name]
                rgb_img, depth_img, cam_name = self.get_camera_data(cam_info)

                Tow = self.get_obj_pose()
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
        import datetime
        move_step = 0.01
        step_count = 4
        space_mat = []
        for z_iter in range(step_count):
            for y_iter in range(step_count):
                for x_iter in range(step_count):
                    space_mat.append([x_iter, y_iter, z_iter])

        # space_mat = (np.array(space_mat) - (step_count/2.0)) * move_step
        space_mat = np.array([0, 0, 0])
        # rot_steps = [
        #     [-0.6698113083839417, -0.7309706807136536, 0.09623405337333679, -0.08818229287862778],
        #     [-0.6220810413360596, -0.6832402348518372, 0.26631492376327515, -0.27436673641204834],
        #     [-0.6477258205413818, -0.6997520327568054, -0.1958569586277008, 0.22900131344795227],
        #     [-0.6385663151741028, -0.6945580244064331, -0.2443033754825592, -0.22392390668392181],
        #     [-0.6698119640350342, -0.7288873791694641, 0.1040114164352417, 0.09623415023088455],
        #     [-0.9658111929893494, -0.2572302222251892, -0.031931549310684204, -0.005196506157517433],
        #     [-0.5010425448417664, -0.8648211359977722, -0.018904540687799454, -0.026253504678606987],
        #     [-0.47907721996307373, -0.8421483039855957, -0.1479398012161255, 0.19847328960895538],
        #     [-0.4967135787010193, -0.8471241593360901, 0.06838780641555786, -0.17602938413619995],
        #     [-0.8783740997314453, -0.4133619964122772, -0.2260165959596634, 0.08071887493133545],
        #     [-0.9027040600776672, -0.41604071855545044, 0.08803550899028778, -0.06552720069885254],
        #              ]

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
        rot_steps = (rot_steps - rot_steps[0])

        rot_init = None
        robot_position, robot_orientation = self.get_robot_pose()

        for step in space_mat:
            rot_init = robot_orientation
            print("Adding step:{}".format(step))
            for rot in rot_steps:
                sim.simxSetObjectPosition(self.clientID, self.robot_handle, -1, robot_position+step, sim.simx_opmode_blocking)
                sim.simxSetObjectQuaternion(self.clientID, self.robot_handle, -1, rot_init+rot, sim.simx_opmode_blocking)
                self.collect_data()
            robot_orientation = rot_init
        self.close()

    def get_obj(self):
        if self.obj_handle is not None:
            return
        sim_ret, self.obj_handle = sim.simxGetObjectHandle(self.clientID, 'link', sim.simx_opmode_blocking)
        sim_ret, self.joint_handle = sim.simxGetObjectHandle(self.clientID, 'UR10_link6_visible', sim.simx_opmode_blocking)

    def get_obj_pose(self):
        if self.obj_handle is None:
            self.get_obj()
        sim_ret, position = sim.simxGetObjectPosition(self.clientID, self.joint_handle, -1, sim.simx_opmode_blocking)
        sim_ret, orientation = sim.simxGetObjectQuaternion(self.clientID, self.joint_handle, -1, sim.simx_opmode_blocking)
        Tow = get_pose_mat(position, orientation)
        return Tow

    def move_obj(self):
        if self.obj_handle is None:
            self.get_obj()

        sim_ret, position = sim.simxGetObjectPosition(self.clientID, self.joint_handle, -1, sim.simx_opmode_blocking)
        sim_ret, orientation = sim.simxGetObjectQuaternion(self.clientID, self.joint_handle, -1, sim.simx_opmode_blocking)

        sim.simxSetObjectPosition(self.clientID, self.obj_handle, -1, position, sim.simx_opmode_blocking)
        sim.simxSetObjectQuaternion(self.clientID, self.obj_handle, -1, orientation, sim.simx_opmode_blocking)

    def get_test_pose(self):
        if self.obj_handle is None:
            self.get_obj()
        sim_ret, position = sim.simxGetObjectPosition(self.clientID, self.obj_handle, -1, sim.simx_opmode_blocking)
        sim_ret, orientation = sim.simxGetObjectQuaternion(self.clientID, self.obj_handle, -1, sim.simx_opmode_blocking)
        return orientation


if __name__ == '__main__':
    ur_sim = VrepRobot('data/vrep_ang', dump_rt=True)
    ur_sim.setup_all()
    ur_sim.run()
    # while(True):
    #     key = input()
    #     print("{}".format(ur_sim.get_test_pose()))
