import sim
import numpy as np


class vrep_robot:
    """ Warper for v-rep robot control"""
    def __init__(self, clientID, robot_target_handler, simx_opmode_blocking):
        self.clientID = clientID
        self.target_handler = robot_target_handler
        self.simx_opmode_blocking = simx_opmode_blocking

    def get_pose(self, use_mat=False):
        """get robot pose """
        sim_ret, robot_position = sim.simxGetObjectPosition(clientID, UR10_target_handle, -1, sim.simx_opmode_blocking)
        sim_ret, robot_orientation = sim.simxGetObjectOrientation(clientID, UR10_target_handle, -1, sim.simx_opmode_blocking)
        if use_mat:
            pose = self._pose_vec2mat(robot_position, robot_orientation)
        else:
            pose = (robot_position, robot_orientation)
        return pose

    def set_pose(self, target_pose):
        """Set robot pose"""
        if len(target_pose) != 2:
            # convert pose from matrix to vector
            target_pose = self._pose_mat2vec(target_pose)

        target_pos, target_rot = target_pose
        sim.simxSetObjectPosition(clientID, self.target_handler, -1, target_pos, self.simx_opmode_blocking)
        sim.simxSetObjectOrientation(clientID, self.target_handler, -1, target_rot, self.simx_opmode_blocking)

    def _pose_mat2vec(self, pose_mat):
        """Convert pose in matrix to vector of position and orientation"""
        pos = []
        rot = []
        return pos, rot

    def _pose_vec2mat(self, pos, rot):
        pose_mat = []
        return pose_mat


if __name__ == '__main__':
    sim.simxFinish(-1) # just in case, close all opened connections
    clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    if clientID != -1:
        print("Connected to remote API server")
        sim_ret, UR10_target_handle = sim.simxGetObjectHandle(clientID, 'UR10_target', sim.simx_opmode_blocking)
        sim_ret, UR10_target_position = sim.simxGetObjectPosition(clientID, UR10_target_handle, -1, sim.simx_opmode_blocking)
        sim_ret, UR10_target_orientation = sim.simxGetObjectOrientation(clientID, UR10_target_handle, -1, sim.simx_opmode_blocking)



        move_z = 0.005
        rot_y = 0.01

        # move the robot
        for step in range(50):
            UR10_target_position[2] += move_z
            UR10_target_orientation[1] += rot_y
            sim.simxSetObjectPosition(clientID, UR10_target_handle, -1, UR10_target_position, sim.simx_opmode_blocking)
            sim.simxSetObjectOrientation(clientID, UR10_target_handle, -1, UR10_target_orientation, sim.simx_opmode_blocking)


        sim_ret, cam_target_handle = sim.simxGetObjectHandle(clientID, 'Vision_sensor',
                                                             sim.simx_opmode_blocking)
        sim_ret, resolution, raw_image = sim.simxGetVisionSensorImage(clientID, cam_target_handle, 0,
                                                                      sim.simx_opmode_blocking)

        color_img = np.asarray(raw_image)


        # TODO: capture robot pose

        # close the simulator
        sim.simxGetPingTime(clientID)
        sim.simxFinish(clientID)

    else:
        print('Failed connecting to remote API server')
    print('Program ended')