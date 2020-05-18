import os
import numpy as np
import tqdm
from PIL import Image
import data_utils
import json
from plyfile import PlyData
from data_utils import get_bbox2d, check_kp_bound


# Hand-craft 3d keypoint definition
kpt3d_ur = np.array([
    # front
    [-0.000354, -0.046325, 0.046288],
    [ 0.024613, -0.014480, 0.040862],
    [-0.024354, -0.015258, 0.041063],
    # left
    [0.044573, -0.030300,  0.005540],
    [0.036552, -0.040553, -0.020870],
    [0.027000, -0.037935, -0.030877],
    # right
    [-0.043479, -0.027063,  0.012919],
    [-0.040687, -0.019993, -0.013801],
])



kpt3d_franka = np.array([
[0.083473, 0.005882, 0.043592],
[0.064259, -0.023191, 0.053959],
[0.089078, 0.007429, 0.006067],
[0.017245, -0.034455, 0.011027],
[-0.053517, -0.027140, 0.027207],
[0.003370, -0.048394, 0.000608],
[0.070592, 0.042233, -0.022286],
[0.089077, 0.006222, -0.026412],
])

kpt3d = kpt3d_franka

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


def record_ann(model_meta, img_id, ann_id, images, annotations, render_mask=True):
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

    rgb_files = os.listdir(rgb_dir)
    rgb_files = [fname for fname in rgb_files if '.png' in fname]
    global vis_iter
    inds = range(len(rgb_files))

    if render_mask:
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

        # Get metadata & annotations
        Tcw = np.array(cam_info[cam_name]['pose'])
        K = np.array(cam_info[cam_name]['intrinsics'])
        Tow = frame_meta['pose']
        pose_Toc = np.dot(np.linalg.inv(Tcw), Tow)[:3]
        corner_2d = data_utils.project(corner_3d, K, pose_Toc)
        center_2d = data_utils.project(center_3d[None], K, pose_Toc)[0]
        fps_2d = data_utils.project(fps_3d, K, pose_Toc)
        box = get_bbox2d(corner_2d, rgb.size[0], rgb.size[1])

        invalid_count = check_kp_bound(fps_2d, rgb.size[0], rgb.size[1])
        if invalid_count > fps_3d.shape[0] * 0.2:
            img_id -= 1
            continue

        mask_path = ''
        if render_mask:
            render_engine = cam_info[cam_name]['render_engine']
            import cv2
            c, d = render_engine.render_depth(pose_Toc, need_color=True, convert=True)
            mask = np.zeros(d.shape)
            mask[np.where(d != 0)] = 255
            mask_path = mask_dir.format(ind)
            cv2.imwrite(mask_path, mask)
        else:
            mask = None

        # vis(rgb, model_2d=None, corner_2d=None, draw_box=False, kpt2d=fps_2d, bbox2d=None, mask=mask)

        ann_id += 1
        anno = {'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'mask_path': mask_path, 'cam_name': cam_name})
        anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose_Toc.tolist()})
        anno.update({'bbox': box, 'type': 'real', 'cls': 'guo'})

        # Append annotation
        images.append(info)
        annotations.append(anno)

    return img_id, ann_id


def _get_vrep_data(data_root, img_id, ann_id, images, annotations, cam_name_list=None, render_mask=True):
    """
    Generate data annotation files for dataset in the given path
    :param data_root: data directory, should include subdirectories [frame, rgb, 'meta.json']
    :param img_id: current image id, will be none zero and matters
        only when generating  one annotation file for more than one data folder
    :param ann_id: current annotation id, will be none zero and matters
        only when generating  one annotation file for more than one data folder
    :param images: array to store image information, will be none empty and matters
        only when generating  one annotation file for more than one data folder
    :param annotations: array to store annotations, will be none empty and matters
        only when generating  one annotation file for more than one data folder
    :param cam_name_list: list of camera names to be used, ['ALL'] means use all cameras available
    :param render_mask: whether or not to render mask with pyrender
    :return:
    """

    # Obtain meta information
    meta = json.load(open(os.path.join(data_root, 'meta.json'), 'r'))
    cam_info = meta['cam_info']

    # remove invalid sensor
    if cam_name_list is not None:
        if cam_name_list[0] == 'ALL':
            cam_name_list = list(cam_info.keys())
        else:
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

    # process the frames to generate annotations and image information
    img_id, ann_id = record_ann(model_meta, img_id, ann_id, images, annotations, render_mask=render_mask)
    return img_id, ann_id


def get_vrep_data(camera_name_list, create_sub_dataset=True, render_mask=True):
    """
    Generate data for VREP
    :param camera_name_list: list of camera names to use.
        set to ['ALL'] if you want to use all camera
    :param create_sub_dataset: whether or not to make data obtained from each camera
        as subdirectory
    :param render_mask: whether or not to render mask with pyrender (dsiable it if pygl is not valid)
    """
    _data_root = 'data'

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    # sequences = [v for v in os.listdir(_data_root) if v.startswith('vrep_franka_link')]
    sequences = ['vrep_franka_val']

    for sequence in sequences:
        data_root = os.path.join(_data_root, sequence)
        img_id, ann_id = _get_vrep_data(data_root, img_id, ann_id, images, annotations,
                                        camera_name_list, render_mask=render_mask)

    # Sort frames from different sensor to separate sub-dataset (by sensor)
    if create_sub_dataset:
        im_anno_dict = dict()
        for anno_idx, anno in enumerate(annotations):
            im_anno_dict[anno['image_id']] = anno_idx

        data_folder = os.path.join(data_root, 'subset_')
        for cam_name in cam_name_list:
            data_folder += '_' + cam_name
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

        np.savetxt(os.path.join(data_folder, 'fps.txt'), kpt3d)
        anno_path = os.path.join(data_folder, 'train.json')

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


def result_analysis(output_path):
    f = open(output_path, 'r')
    lines = f.readlines()
    f.close()
    init = False
    line_id = 0
    train_dict = {
        'vote_loss':[],
        'seg_loss':[],
        'loss': []
    }
    val_dict = {
        'epoch':[],
        'vote_loss':[],
        'seg_loss': [],
        'loss': [],
        'add': [],
        '2dproj': [],
        'mask_ap': [],
    }
    while(True):
        line = lines[line_id]
        if 'epoch' in line and not init:
            init = True

        if init:
            if 'epoch' in line:
                data = line.split(' ')
                take_data = False
                for id, item in enumerate(data):
                    if 'vote_loss' in item:
                        train_dict['vote_loss'].append(float(data[id+1]))
                    elif 'seg_loss' in item:
                        train_dict['seg_loss'].append(float(data[id+1]))
                    elif 'loss' in item:
                        train_dict['loss'].append(float(data[id+1]))
            else:
                curr_epoch = len(train_dict['loss']) - 1
                val_dict['epoch'].append(curr_epoch)
                data = line.split('\'')
                for id, item in enumerate(data):
                    if 'vote_loss' in item:
                        loss_val = float(item.split(':')[-1])
                        val_dict['vote_loss'].append(loss_val)
                    elif 'seg_loss' in item:
                        loss_val = float(item.split(':')[-1])
                        val_dict['seg_loss'].append(loss_val)
                    elif 'loss' in item:
                        loss_val = float(item.split(':')[-1])
                        val_dict['loss'].append(loss_val)
                for i in range(1, 5):
                    line = lines[line_id + i]
                    if i == 1:
                        val_dict['2dproj'].append(float(line.split(':')[1]))
                    elif i == 2:
                        val_dict['add'].append(float(line.split(':')[1]))
                    elif i == 4:
                        val_dict['mask_ap'].append(float(line.split(':')[1]))
                line_id += i
        line_id += 1
        if line_id >= len(lines):
            break

    # for train: train loss vs. val loss

    # for val: change of metrics
    # plot result
    from matplotlib import pyplot as plt
    save_dir = 'data/train_img.pdf'

    SMALL_SIZE = 7
    BIGGER_SIZE = 10
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # plt.plot(time_rcnn_baseline, ap_baseline, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12, label='rcnn_baseline')
    # style_label = 'fivethirtyeight'
    style_label = 'seaborn-dark'
    style_label = 'seaborn-colorblind'
    with plt.style.context(style_label):
        # if True:
        # plt.figure(figsize=(12, 6))
        # plt.xlim(100, 45)
        plt.ylim(0, 0.1)
        # TODO: adjust here
        train_epoch = list(range(len(train_dict['vote_loss'])))
        plt.plot(train_epoch, train_dict['vote_loss'], color='lightskyblue', linestyle='dashed', linewidth=2,
                 label='heatmap_loss(train)')
        plt.plot(train_epoch, train_dict['seg_loss'], color='turquoise', linestyle='dashed', linewidth=1,
                 label='seg_loss(train)')

        plt.plot(val_dict['epoch'], val_dict['vote_loss'], color='tomato', linestyle='dashed', linewidth=2,
                 label='heatmap_loss(val)')
        plt.plot(val_dict['epoch'], val_dict['seg_loss'], color='sandybrown', linestyle='dashed', linewidth=1,
                 label='seg_loss(val)')

        # plt.plot(time_overall_baseline, ap_baseline_hard, color='darkviolet', marker='x', linestyle='dashed', linewidth=2, markersize=12, label='PRCNN hard')
        # plt.plot(time_overall_ours, ap_ours_hard, color='darkslateblue', marker='x', linestyle='dashed', linewidth=2, markersize=12, label='ours hard')

        # plt.legend(prop={'size': 12})
        plt.legend()
        plt.xlabel('Number of Epoch')
        plt.ylabel('Loss Value')
        plt.axis('on')
        plt.grid(b=True)
        plt.rc('axes', titlesize=30)
        plt.tight_layout()
    plt.savefig(save_dir)
    plt.show()



    save_dir = 'data/eval_metrics.pdf'

    SMALL_SIZE = 7
    BIGGER_SIZE = 10
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # plt.plot(time_rcnn_baseline, ap_baseline, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12, label='rcnn_baseline')
    # style_label = 'fivethirtyeight'
    style_label = 'seaborn-dark'
    style_label = 'seaborn-colorblind'
    with plt.style.context(style_label):
        # if True:
        # plt.figure(figsize=(12, 6))
        # plt.xlim(100, 45)
        # plt.ylim(0, 0.1)
        # TODO: adjust here

        plt.plot(val_dict['epoch'], val_dict['add'], color='lightskyblue', linestyle='dashed', linewidth=1.5,
                 label='ADD')
        plt.plot(val_dict['epoch'], val_dict['2dproj'], color='turquoise', linestyle='dashed', linewidth=1.5,
                 label='2D Proj')

        # plt.plot(val_dict['epoch'], val_dict['mask_ap'], color='darkslateblue', linestyle='dashed', linewidth=1.5,
        #          label='Mask AP')

        plt.legend()
        plt.xlabel('Number of Epoch')
        plt.ylabel('Precision')
        plt.axis('on')
        plt.grid(b=True)
        plt.rc('axes', titlesize=30)
        plt.tight_layout()
    plt.savefig(save_dir)
    plt.show()


import cv2
import os
import numpy as np


click_record = []

IM_DICT = {
    1: (380, 460),
    2: (1160, 1180),
    3: (1390, 1440),
    4: (1960, 2020),
    5: (2410, 2450),
    6: (2780, 2900),
    7: (3200, 3220),
    8: (3540, 3580),
    9: (3910, 3940),
    10: (4260, 4280),
    11: (4610, 4630),
    12: (4840, 4910)
}


MODEL_DICT = {
    1: 3,
    2: 9,
    3: 5,
    4: -5,
    5: -6.1,
    6: 8,
    7: -4.1,
    8: 9.1,
    9: 8.1,
    10: 5.1,
    11: 4.1,
    12: 2.1
}


IM_RANGE = ([640, 108], [1279, 587])

vis_iter = 0

def output_video(im_list, im_shape):
    output_path = "./out.mp4"
    height, width, layers = im_shape

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    from tqdm import tqdm
    for i in tqdm(range(len(im_list))):
        # if i % 100 == 0:
        #     print(f"Writting:{i}")
        out.write(im_list[i])
    out.release()


def dict_to_list():
    fid_pairs = IM_DICT.values()
    fid_list = []
    for val in fid_pairs:
        # fid_list.append(int(val[0]))
        fid_list.append(int(val[1]))
    return fid_list


def dict_to_range():
    fid_pairs = IM_DICT.values()
    fid_list = []
    for val in fid_pairs:
        fid_list.append(val)
    return fid_list


def in_range(range_dict, fid):
    for fid_range in range_dict:
        if fid >= fid_range[0] and fid <= fid_range[1]:
            return True
    return False


def mouse_callback(event, x, y, flags, param):
    global click_record
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f"[{x}, {y}]")
        click_record.append([x, y])
        click_rc = np.array(click_record)
        if len(click_rc) > 1:
            print(f"Shape: {click_rc[-1] - click_rc[-2]}")


def load_process():
    video_path = "input.mkv"
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    im_shape = None
    im_list = []
    id_list = dict_to_list()
    range_dict = dict_to_range()
    output_dir = './out_imgs'

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    vis_iter = 0
    while(cap.isOpened()):

        if frame_id % 50 == 0:
            print(f"Processing frame:{frame_id}")

        frame_id += 1
        ret, img = cap.read()

        if ret == False or img is None:
            break

        if frame_id not in id_list:
            continue

        # if frame_id % 10 != 0:
        #     continue

        text = "ID:" + str(frame_id)

        im_y0 = 108
        im_y1 = 587
        im_x0 = 640
        im_x1 = 1279

        img_crop, catch_success = detector(img[im_y0:im_y1, im_x0:im_x1,:])
        vis_path = os.path.join(output_dir, '{}.png'.format(vis_iter))
        cv2.imwrite(vis_path, img_crop)
        vis_iter += 1
        img[im_y0:im_y1, im_x0:im_x1, :] = img_crop
        im_shape = img.shape
        im_list.append(img)

    # output_video(im_list, im_shape)


def detector(img):
    # RPN stage: Get region proposal with pre-defined parameters
    x_margin = 10
    y_margin = 10
    x_size = 200
    y_size = 200
    height, width, layers = img.shape
    x0, y0, x1, y1 = bbox_from_shape(width, height)

    # Draw the original red box
    box_pts = [
        [(x0, y0), (x1, y0)],
        [(x1, y0), (x1, y1)],
        [(x1, y1), (x0, y1)],
        [(x0, y1), (x0, y0)]
    ]

    for pts in box_pts:
        pt1, pt2 = pts
        cv2.line(img, pt1, pt2, (0, 0, 255), thickness=1)

    x0 = x0 + x_margin
    x1 = x0 + x_size
    y0 = y0 + y_margin
    y1 = y0 + y_size

    # Crop RoI with region proposals, apply canny's edge detector
    im_crop = img[y0:y1, x0:x1, :]
    gaus = cv2.GaussianBlur(im_crop, (3, 3), 0)
    gray = cv2.cvtColor(gaus, cv2.COLOR_BGR2GRAY)
    gradx = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    grady = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    edge_out = cv2.Canny(gradx, grady, 50, 150)

    # RCNN stage: refine bounding box, produce objectness socre
    bbox = detection_head(edge_out)

    # Draw overlaid bounding box
    if bbox is not None:
        box_pts = [
            [(x0, y0), (x1, y0)],
            [(x1, y0), (x1, y1)],
            [(x1, y1), (x0, y1)],
            [(x0, y1), (x0, y0)]
        ]

        # for pts in box_pts:
        #     pt1, pt2 = pts
        #     cv2.line(img, pt1, pt2, (255, 0, 0), thickness=2)

        x_offset = x0
        y_offset = y0
        x0, y0, x1, y1 = bbox
        x0 = x0 + x_offset
        x1 = x1 + x_offset
        y0 = y0 + y_offset
        y1 = y1 + y_offset

        box_pts = [
            [(x0, y0), (x1, y0)],
            [(x1, y0), (x1, y1)],
            [(x1, y1), (x0, y1)],
            [(x0, y1), (x0, y0)]
        ]

        for pts in box_pts:
            pt1, pt2 = pts
            cv2.line(img, pt1, pt2, (0, 255, 0), thickness=2)
        return img, True
    else:
        box_pts = [
            [(x0, y0), (x1, y0)],
            [(x1, y0), (x1, y1)],
            [(x1, y1), (x0, y1)],
            [(x0, y1), (x0, y0)]
        ]

        for pts in box_pts:
            pt1, pt2 = pts
            cv2.line(img, pt1, pt2, (0, 255, 0), thickness=2)

        return img, False

def procecss():
    im_list = os.listdir('data/vis')
    im_list = [i for i in im_list if '.png' in i]
    im_list.sort()
    img_list = []
    from tqdm import tqdm
    for im_id in range(len(im_list)):
        im_path = str(im_id) + '.png'
        im_path = os.path.join('data/vis', im_path)
        img = cv2.imread(im_path)
        # if im_id % 50 == 0:
        if True:
            print(f"Processing frame:{im_id}")

            img = img[61:423, 85:570, :]

            # cv2.imshow("Annotate", img)
            # key = cv2.waitKey(0)

            img_list.append(img)
            im_shape = img.shape
            out_path = os.path.join('data/vis_clean', str(im_path))
            cv2.imwrite(out_path, img)

    print(f"processing num images:{len(img_list)}")

    output_video(img_list, im_shape)


if __name__ == "__main__":
    cam_name_list = ['ALL']
    get_vrep_data(cam_name_list, create_sub_dataset=False, render_mask=True)
