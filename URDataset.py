import os
import numpy as np
import tqdm
from PIL import Image
import data_utils
import json
from plyfile import PlyData
from data_utils import get_bbox2d


# Hand-craft 3d keypoint definition
kpt3d = np.array([
    # front
    [-0.000354, -0.046325, 0.046288],
    # [ 0.000273, -0.012989, 0.049855],
    [ 0.024613, -0.014480, 0.040862],
    [-0.024354, -0.015258, 0.041063],
    # left
    # [0.044108, -0.000702,  0.000098],
    [0.044573, -0.030300,  0.005540],
    [0.036552, -0.040553, -0.020870],
    [0.027000, -0.037935, -0.030877],
    # right
    # [-0.044183, -0.005127, -0.000864],
    [-0.043479, -0.027063,  0.012919],
    [-0.040687, -0.019993, -0.013801],
    # [-0.040995, -0.037231, -0.012979],
])

vis_iter = 0


def vis(rgb, model_2d=None, corner_2d=None, draw_box=False, kpt2d=None, bbox2d=None, mask=None):
    global vis_iter
    from matplotlib import pyplot as plt
    import copy

    plt.close()
    rgb_vis = np.array(rgb)
    fg_id = np.where(mask != 0)
    rgb_vis[fg_id[0], fg_id[1], :] = 255
    plt.imshow(rgb_vis)

    if model_2d is not None:
        plt.plot(model_2d[:, 0], model_2d[:, 1], 'b.')

    if corner_2d is not None:
        plt.plot(corner_2d[:, 0], corner_2d[:, 1], 'r.')
        if draw_box:
            from data_utils import draw_box3d
            draw_box3d(corner_2d, plt=plt)

    if kpt2d is not None:
        plt.plot(kpt2d[:, 0], kpt2d[:, 1], 'g.')

    if bbox2d is not None:
        from data_utils import draw_box2d
        draw_box2d(bbox2d, plt=plt)

    plt.savefig('data/vis/{}.png'.format(vis_iter))
    # plt.show()
    vis_iter += 1


def read_ply_points(ply_path):
    ply = PlyData.read(ply_path)
    data = ply.elements[0].data
    points = np.stack([data['x'], data['y'], data['z']], axis=1)
    return points


def sample_fps_points(data_root):
    ply_path = os.path.join(data_root, 'model.ply')
    ply_points = read_ply_points(ply_path)
    fps_points = fps_utils.farthest_point_sampling(ply_points, 8, True)
    np.savetxt(os.path.join(data_root, 'fps.txt'), fps_points)


def get_model_corners(model):
    min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
    min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
    min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def record_ann(model_meta, img_id, ann_id, images, annotations, generate_mask=True):
    data_root = model_meta['data_root']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']
    cam_info = model_meta['cam_info']
    meta_dir = os.path.join(data_root, 'frame')
    rgb_dir = os.path.join(data_root, 'rgb')
    mask_dir = os.path.join(data_root, 'mask', '{}_mask.png')

    if not os.path.isdir(os.path.dirname(mask_dir)):
        os.makedirs(os.path.dirname(mask_dir))

    inds = range(len(os.listdir(rgb_dir)))

    if generate_mask:
        from render_utils import RenderingEngine
        for cam_name in cam_info.keys():
            model_path = model_meta['model_path']
            K = np.array(cam_info[cam_name]['intrinsics'])
            im_shape = cam_info[cam_name]['im_shape']
            render_engine = RenderingEngine(K, im_shape, model_path)
            cam_info[cam_name].update({'render_engine': render_engine})
    else:
        for cam_name in cam_info.keys():
            cam_info[cam_name].update({'render_engine': None})

    for ind in tqdm.tqdm(inds):

        # Load metadata
        meta_path = os.path.join(meta_dir, '{}.json'.format(ind))
        frame_meta = json.load(open(meta_path, 'r'))
        cam_name = frame_meta['cam_name']
        # skip frame from invalid camera
        if cam_name not in cam_info.keys():
            continue

        # load rgb image
        rgb_path = os.path.join(rgb_dir, '{}.png'.format(ind))
        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        # Get metadata & annotations
        Tcw = np.array(cam_info[cam_name]['pose'])
        K = np.array(cam_info[cam_name]['intrinsics'])
        Tow = frame_meta['pose']
        pose_Toc = np.dot(np.linalg.inv(Tcw), Tow)[:3]
        corner_2d = data_utils.project(corner_3d, K, pose_Toc)
        center_2d = data_utils.project(center_3d[None], K, pose_Toc)[0]
        fps_2d = data_utils.project(fps_3d, K, pose_Toc)
        box = get_bbox2d(corner_2d, rgb.size[0], rgb.size[1])

        mask_path = ''
        if generate_mask:
            render_engine = cam_info[cam_name]['render_engine']
            import cv2
            c, d = render_engine.render_depth(pose_Toc, need_color=True, convert=True)
            mask = np.zeros(d.shape)
            mask[np.where(d != 0)] = 255
            mask_path = mask_dir.format(ind)
            cv2.imwrite(mask_path, mask)

        vis(rgb, model_2d=None, corner_2d=None, draw_box=False, kpt2d=fps_2d, bbox2d=None, mask=mask)

        ann_id += 1
        anno = {'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'mask_path': mask_path, 'cam_name': cam_name})
        anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose_Toc.tolist()})
        anno.update({'bbox': box, 'type': 'real', 'cls': 'guo'})
        annotations.append(anno)

    return img_id, ann_id


def _get_vrep_data(data_root, img_id, ann_id, images, annotations, cam_name_list=None):
    meta = json.load(open(os.path.join(data_root, 'meta.json'), 'r'))

    cam_info = meta['cam_info']

    # remove invalid sensor
    if cam_name_list is not None:
        pop_list = []
        for cam_name in cam_info.keys():
            if cam_name not in cam_name_list:
                pop_list.append(cam_name)
        for cam_name in pop_list:
            cam_info.pop(cam_name)

    model = read_ply_points(os.path.join(data_root, 'model.ply'))
    model_path = os.path.join(data_root, 'model.ply')
    corner_3d = get_model_corners(model)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    fps_3d = kpt3d
    model_meta = {
        'cam_info': cam_info,
        'corner_3d': corner_3d,
        'center_3d': center_3d,
        'fps_3d': fps_3d,
        'data_root': data_root,
        'model': model,
        'model_path': model_path,
    }

    img_id, ann_id = record_ann(model_meta, img_id, ann_id, images, annotations)
    return img_id, ann_id


def get_vrep_data(camera_name_list, create_sub_dataset=True):
    _data_root = 'data'

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    sequences = [v for v in os.listdir(_data_root) if v.startswith('custom')]

    for sequence in sequences:
        data_root = os.path.join(_data_root, sequence)
        img_id, ann_id = _get_vrep_data(data_root, img_id, ann_id, images, annotations, camera_name_list)

    if create_sub_dataset:
        im_anno_dict = dict()
        for anno_idx, anno in enumerate(annotations):
            im_anno_dict[anno['image_id']] = anno_idx

        data_folder = os.path.join(data_root, 'sub_dataset')

        for cam_name in cam_name_list:
            data_folder += '_' + cam_name[-1]
        if os.path.isdir(data_folder):
            os.system('rm -r {}'.format(data_folder))
        os.system('mkdir -p {}'.format(data_folder))

        if os.path.isfile(os.path.join(data_root, 'model.ply')):
            os.system('cp {} {}'.format(os.path.join(data_root, 'model.ply'), os.path.join(data_folder, 'model.ply')))

        init_checked = False
        for im_anno in images:
            rgb_path = im_anno['file_name']
            im_id = im_anno['id']
            mask_path = annotations[im_anno_dict[im_id]]['mask_path']

            rgb_out_path = rgb_path.replace(data_root, data_folder)
            mask_out_path = mask_path.replace(data_root, data_folder)

            # create folder
            if not init_checked:
                for data_output_path in [rgb_out_path, mask_out_path]:
                    if os.path.isdir(os.path.dirname(data_output_path)):
                        os.system('rm -r {}'.format(os.path.dirname(data_output_path)))
                    os.system('mkdir -p {}'.format(os.path.dirname(data_output_path)))
                init_checked = True

            os.system('cp {} {}'.format(rgb_path, rgb_out_path))
            os.system('cp {} {}'.format(mask_path, mask_out_path))
            # im_anno['file_name'] = rgb_out_path
            # annotations[im_anno_dict[im_id]]['mask_path'] = mask_out_path


        np.savetxt(os.path.join(data_folder, 'fps.txt'), kpt3d)
        anno_path = anno_path = os.path.join(data_folder, 'train.json')

        categories = [{'supercategory': 'none', 'id': 1, 'name': 'ur'}]
        instance = {'images': images, 'annotations': annotations, 'categories': categories}
        with open(anno_path, 'w') as f:
            json.dump(instance, f, indent=4)

    else:
        anno_path = os.path.join(data_root, 'train.json')
        np.savetxt(os.path.join(data_root, 'fps.txt'), kpt3d)
        categories = [{'supercategory': 'none', 'id': 1, 'name': 'ur'}]
        instance = {'images': images, 'annotations': annotations, 'categories': categories}
        with open(anno_path, 'w') as f:
            json.dump(instance, f, indent=4)


if __name__ == "__main__":
    cam_name_list = ['Vision_sensor']
    get_vrep_data(cam_name_list)
