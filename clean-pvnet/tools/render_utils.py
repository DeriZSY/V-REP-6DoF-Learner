import copy
import os

import numpy as np
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
import trimesh


def pose_cv2cg(T):
    """
    Convert pose from CV coordinate to CG coordinate
    T: [4, 4]
    :param T:
    :return:
    """
    R = T[:3, :3]
    t = T[:3, 3]

    R_rot = np.eye(3)
    R_rot[1, 1] = -1
    R_rot[2, 2] = -1
    R = np.matmul(R_rot, R)
    t = np.matmul(R_rot, t)

    T_cg = np.eye(4)
    T_cg[:3, :3] = R
    T_cg[:3, 3] = t
    return T_cg


class RenderingEngine(object):
    """ Class to handle model rendering with pyrender (non differenciable )"""

    def __init__(self, K, img_shape, model=None):

        # setup basic parameters
        if model is not None and isinstance(model, trimesh.base.Trimesh):
            self.model = pyrender.Mesh.from_trimesh(model)
        elif model is not None and isinstance(model,
                                              str) and os.path.isfile(model):
            self.model = pyrender.Mesh.from_trimesh(trimesh.load(model))
        else:
            self.model = None

        self.img_size = img_shape

        # self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.K = K

        self.scene = pyrender.Scene()
        camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
        light = pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0], intensity=20.0)
        self.scene.add(camera, pose=np.eye(4))
        self.scene.add(light, pose=np.eye(4))

    def render_depth(self, obj_pose, need_color=False, convert=False):
        assert self.model is not None, "Must set at least one model before rendering"
        if convert:
            obj_pose = pose_cv2cg(obj_pose)

        curr_scene = copy.deepcopy(self.scene)
        curr_scene.add(self.model, pose=obj_pose)
        # apply rendering
        r = pyrender.OffscreenRenderer(self.img_size[1], self.img_size[0])
        c, depth = r.render(curr_scene)
        if need_color:
            return c, depth
        else:
            return depth

    def set_model(self, model_trimesh):
        assert (isinstance(
            model_trimesh,
            trimesh.base.Trimesh)), "Error: input model is not a trimesh model"
        self.model = pyrender.Mesh.from_trimesh(model_trimesh)

    @staticmethod
    def depth_normalize(depth, use_color=False):
        import cv2
        depth = np.uint8(depth)
        depth = cv2.equalizeHist(depth)
        depth_val = np.unique(depth)
        near = depth_val[1]
        far = depth_val[-1]

        depth_norm = (depth - near) / (far - near) * 255
        depth_norm[np.where(depth_norm >= 255)] = 255
        depth_norm[np.where(depth_norm <= 0)] = 255

        depth_norm = np.uint8(depth_norm)
        if use_color:
            depth_norm = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        return depth_norm

    @staticmethod
    def inv_projection(depth):
        """Inverse projection depth with camera intrinsic. P = K.I * p * D"""

        # camera intrinsic
        K = np.array([[650.048, 0, 324.328], [0, 647.183, 257.323],
                      [0, 0, 1]])  # FIXME: Should be loaded from config
        K_I = np.linalg.inv(K)

        _, size_x, size_y = depth.shape

        # construct 2*n array including coordinate from [0,0] to [size_x-1, size_y-1]
        p = np.mgrid[0:size_x:1, 0:size_y:1].reshape(2, -1)
        p_H = np.vstack((p, np.ones(size_x * size_y)))  # shape: 3 * n

        depth_vec = np.squeeze(depth.reshape(1, -1))
        P = np.matmul(K_I, p_H)
        # remove background points
        fg_idx = np.where(depth_vec != 0)[0]
        P = P[:, fg_idx]
        depth_vec = depth_vec[fg_idx]
        P = np.transpose(
            np.multiply(P, np.vstack(
                (depth_vec, depth_vec,
                 depth_vec))))  # n * 3 in camera coordinate
        return P
