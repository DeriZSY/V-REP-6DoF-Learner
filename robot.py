import sim
import numpy as np
import utils
import os


class VrepRobot:
    """ Warper for v-rep robot control"""
    def __init__(self, data_root, dump_rt=True):
        sim.simxFinish(-1)  # Just in case, close all opened connections
        self.robot_dicts = {}
        self.camera_dicts = {}
        self.dump_rt = dump_rt
        self.frame_id = 0

        self.robot_name = 'UR10_target'
        self.cam_names = ['Vision_sensor', 'Vision_sensor0']
        self.default_cam = 'Vision_sensor'

        self.clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
        assert self.clientID != -1, "Failed to connect to simulation server."

        assert os.path.isdir(os.path.dirname(data_root))
        if not os.path.isdir(data_root):
            os.makedirs(data_root)
        self.data_root = data_root
        self.im_path = os.path.join(self.data_root, 'rgb', 'rgb_{}.png')
        self.depth_path = os.path.join(self.data_root, 'depth', 'depth_{}')
        self.frame_info_path = os.path.join(self.data_root, 'frame', 'frame_{}.json')
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
        # FIXME: whether need to flip?
        # color_img = np.fliplr(color_img)
        color_img = np.flipud(color_img)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = sim.simxGetVisionSensorDepthBuffer(self.clientID, cam_handle,
                                                                               sim.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        # FIXME: whether need to flip?
        depth_img.shape = (resolution[1], resolution[0])
        # depth_img = np.fliplr(depth_img)
        depth_img = np.flipud(depth_img)
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
        if cam_names is None:
            cam_names = self.cam_names

        if cam_names is None or len(cam_names) == 0:
            print("No camera to add, exiting ...")

        for cam_name in cam_names:
            sim_ret, cam_handle = sim.simxGetObjectHandle(self.clientID, cam_name,
                                                          sim.simx_opmode_blocking)
            # Get camera pose and intrinsics in simulation
            sim_ret, cam_position = sim.simxGetObjectPosition(self.clientID, cam_handle, -1,
                                                              sim.simx_opmode_blocking)
            sim_ret, cam_orientation = sim.simxGetObjectOrientation(self.clientID, cam_handle, -1,
                                                                    sim.simx_opmode_blocking)
            cam_trans = np.eye(4, 4)
            cam_trans[0:3, 3] = np.asarray(cam_position)
            cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
            cam_rotm = np.eye(4, 4)

            cam_rotm[0:3, 0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))

            cam_pose = np.dot(cam_trans, cam_rotm)  # Compute rigid transformation representating camera pose
            cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
            cam_depth_scale = 1
            cam_info_dict = {
                'name': cam_name,
                'handle': cam_handle,
                'pose': cam_pose.tolist(),
                'intrinsics': cam_intrinsics.tolist(),
                'depth_scale': cam_depth_scale
            }
            self.camera_dicts[cam_name] = cam_info_dict

    def get_robot_pose(self):
        assert self.robot_handle is not None, "Robot handler isn't set"
        sim_ret, position = sim.simxGetObjectPosition(self.clientID, self.robot_handle, -1, sim.simx_opmode_blocking)
        sim_ret, orientation = sim.simxGetObjectOrientation(self.clientID, self.robot_handle, -1, sim.simx_opmode_blocking)
        # parse to matrix if needed
        pose = (position, orientation)
        return pose

    def set_pose(self, target_pose):
        assert self.robot_handle is not None, "Robot handler isn't set"
        if len(target_pose) != 2 and not isinstance(target_pose, tuple):
            raise NotImplementedError("Only original V-REP format is allowed for robot control at present")

        target_pos, target_rot = target_pose
        sim.simxSetObjectPosition(self.clientID, self.robot_handle, -1, target_pos, sim.simx_opmode_blocking)
        sim.simxSetObjectOrientation(self.clientID, self.robot_handle, -1, target_rot, sim.simx_opmode_blocking)

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
            'cam_info': cams_info,
            'robot_info': self.robot_handle,
            'time': timestamp
        }
        self.meta_data = meta_data

    def collect_data(self):
        rgb_img, depth_img, cam_name = self.get_camera_data()
        pos, rot = self.get_robot_pose()

        rgb_path = self.im_path.format(self.frame_id)
        depth_path = self.depth_path.format(self.frame_id)
        info_path = self.frame_info_path.format(self.frame_id)

        frame_data_dict = {
            'frame_id': self.frame_id,
            'cam_name': cam_name,
            'robot_pos': pos,
            'robot_rot': rot,
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

    def dump_data(self):
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

            # Save depth
            im_depth = frame_info['im_depth']
            depth_path = frame_info['depth']
            frame_info.pop('im_depth')
            utils.save_depth(im_depth, depth_path)

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
        move_z = -0.005
        rot_y = 0
        for step in range(3):
            print("start moving:{}".format(datetime.datetime.now().timestamp()))
            robot_position, robot_orientation = self.get_robot_pose()
            robot_position[2] += move_z
            robot_orientation[1] += rot_y
            sim.simxSetObjectPosition(self.clientID, self.robot_handle, -1, robot_position, sim.simx_opmode_blocking)
            sim.simxSetObjectOrientation(self.clientID, self.robot_handle, -1, robot_orientation, sim.simx_opmode_blocking)
            self.collect_data()
        self.close()


if __name__ == '__main__':
    ur_sim = VrepRobot('/Users/DeriZSY/vrep_data2', dump_rt=False)
    ur_sim.setup_all()
    ur_sim.run()
