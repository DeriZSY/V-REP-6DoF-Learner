import pickle
import os
import numpy as np


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def get_pose_mat(position, orientation):
    from transforms3d.quaternions import quat2mat
    x, y, z, w = orientation
    quat_new = [w, x, y, z]
    mat_R = quat2mat(quat_new)
    pose = np.eye(4, 4)
    pose[:3, :3] = mat_R
    pose[:3, 3] = position
    return pose


def draw_box2d(box2d, image=None, plt=None):
    x_min, y_min, x_max, y_max = box2d

    points = [
        [x_min, y_min],
        [x_min, y_max],
        [x_max, y_max],
        [x_max, y_min]
    ]
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]
    ]

    for id, line in enumerate(lines):
        pt1 = points[line[0]]
        pt2 = points[line[1]]

        if image is not None:
            import cv2
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0]), int(pt2[1]))
            cv2.line(image, pt1, pt2, [0, 0, 255], 3)

        if plt is not None:
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])


def draw_box3d(box3d_proj, image=None, plt=None):
    lines = [
        [1, 2],
        [1, 3],
        [1, 5],
        [2, 6],
        [2, 4],
        [4, 3],
        [4, 8],
        [8, 6],
        [8, 7],
        [5, 7],
        [3, 7],
        [5, 6]
    ]

    # orange, blue, green, purple red, pink, caffe
    color_list = [(47, 92, 247), (68, 27, 203), (221, 168, 81), (62, 129, 27), (129, 51, 111), (27, 177, 255),
                  (56, 67, 115), (47, 92, 247), (68, 27, 203), (221, 168, 81), (62, 129, 27), (27, 177, 255)]
    for id, line in enumerate(lines):
        pt1 = box3d_proj[line[0] - 1, :]
        pt2 = box3d_proj[line[1] - 1, :]

        if image is not None:
            import cv2
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0]), int(pt2[1]))
            cv2.line(image, pt1, pt2, color_list[id], 3)

        if plt is not None:
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])


def get_bbox2d(corner2d, w, h):
    x_max = int(min(np.max(corner2d[:, 0]), w))
    x_min = int(max(np.min(corner2d[:, 0]), 0))
    y_max = int(min(np.max(corner2d[:, 1]), h))
    y_min = int(max(np.min(corner2d[:, 1]), 0))
    return [x_min, y_min, x_max, y_max]
