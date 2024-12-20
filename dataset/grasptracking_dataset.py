import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image

import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, get_workspace_mask


class GraspTracking_Dataset(Dataset):
    def __init__(self, root, valid_obj_idxs=None, grasp_labels=None, camera='kinect', split='train', num_points=20000,
                 sequence_size=15, remove_outlier=True, remove_invisible=True, augment=False, load_label=True):
        assert (num_points <= 50000)
        self.root = root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.valid_obj_idxs = valid_obj_idxs
        self.grasp_labels = grasp_labels
        self.sequence_size = sequence_size
        self.camera = camera
        self.augment = augment
        self.obj_num = 89
        self.load_label = load_label
        self.collision_labels = {}

        if split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.colorpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.sequence_number = []
        self.graspnesspath = []
        if self.split == 'train':
            for scene_id in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
                for img_id in range(1, 256-self.sequence_size+1):
                    self.colorpath.append([os.path.join(root, 'scenes', scene_id, camera, 'rgb', str(i).zfill(4) + '.png')
                                            for i in range(img_id, img_id+self.sequence_size)])
                    self.depthpath.append([os.path.join(root, 'scenes', scene_id, camera, 'depth', str(i).zfill(4) + '.png')
                                            for i in range(img_id, img_id+self.sequence_size)])
                    self.labelpath.append([os.path.join(root, 'scenes', scene_id, camera, 'label', str(i).zfill(4) + '.png')
                                            for i in range(img_id, img_id+self.sequence_size)])
                    self.metapath.append([os.path.join(root, 'scenes', scene_id, camera, 'meta', str(i).zfill(4) + '.mat')
                                            for i in range(img_id, img_id+self.sequence_size)])
                    self.scenename.append(scene_id.strip())
                    self.frameid.append([i for i in range(img_id, img_id+self.sequence_size)])
                    self.sequence_number.append(img_id)
        else:
            for scene_id in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
                for img_id in range(int(255/15)):
                    self.colorpath.append([os.path.join(root, 'scenes', scene_id, camera, 'rgb', str(i).zfill(4) + '.png')
                                            for i in range(img_id*15+1, img_id*15+self.sequence_size+1)])
                    self.depthpath.append([os.path.join(root, 'scenes', scene_id, camera, 'depth', str(i).zfill(4) + '.png')
                                            for i in range(img_id*15+1, img_id*15+self.sequence_size+1)])
                    self.labelpath.append([os.path.join(root, 'scenes', scene_id, camera, 'label', str(i).zfill(4) + '.png')
                                            for i in range(img_id*15+1, img_id*15+self.sequence_size+1)])
                    self.metapath.append([os.path.join(root, 'scenes', scene_id, camera, 'meta', str(i).zfill(4) + '.mat')
                                            for i in range(img_id*15+1, img_id*15+self.sequence_size+1)])
                    self.scenename.append(scene_id.strip())
                    self.frameid.append([i for i in range(img_id*15+1, img_id*15+self.sequence_size+1)])
                    self.sequence_number.append(img_id)

    def scene_list(self):
        return self.scenename
    
    def sequence_list(self):
        return self.sequence_number

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list, random_x, random_z):

        # Rotation along x-axis
        rot_angle = (random_x * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along z-axis
        rot_angle = (random_z * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[c, -s, 0],
                            [s, c, 0],
                            [0, 0, 1]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)
            
        return point_clouds, object_poses_list

    def __getitem__(self, index):
        end_points = {}
        for i in range(self.sequence_size):
            if self.load_label:
                end_points['frame_{}'.format(str(i))] = self.get_data_label(index, i)
            else:
                end_points['frame_{}'.format(str(i))] = self.get_data(index, i)
        return end_points

    def get_data(self, index, frame, return_raw_cloud=False):
        color = np.array(Image.open(self.colorpath[index][frame]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index][frame]))
        seg = np.array(Image.open(self.labelpath[index][frame]))
        meta = scio.loadmat(self.metapath[index][frame])
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index][frame]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]

        if return_raw_cloud:
            return cloud_masked
        # sample points random
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]

        init_poses = np.concatenate([np.eye(3), np.array([0,0,0])[:, None]], axis=1)
        object_poses = np.repeat(init_poses[None, ...], self.obj_num, axis=0)   # (89, 3, 4)

        for i, obj_idx in enumerate(obj_idxs):
            object_poses[obj_idx, :, :] = poses[:, :, i]
        
        tmp = np.array([0,0,0,1])[None, None, :]
        tmp = np.repeat(tmp, self.obj_num, axis=0)
        object_poses = np.concatenate([object_poses, tmp], axis=1)

        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['cloud_segs'] = seg_sampled.astype(np.int64)
        ret_dict['frame_id'] = np.array(self.frameid[index][frame]).astype(np.int64)
        ret_dict['camera_pose'] = camera_poses[self.frameid[index][frame]].astype(np.float32)
        ret_dict['scene_name'] = np.array(int(self.scenename[index].split('_')[1])).astype(np.int64)
        ret_dict['object_poses'] = object_poses.astype(np.float32)

        return ret_dict

    def get_data_label(self, index, frame):
        color = np.array(Image.open(self.colorpath[index][frame]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index][frame]))
        seg = np.array(Image.open(self.labelpath[index][frame]))
        meta = scio.loadmat(self.metapath[index][frame])
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index][frame]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]
        objectness_label = seg_sampled.copy()

        objectness_label[objectness_label > 1] = 1

        init_poses = np.concatenate([np.eye(3), np.array([0,0,0])[:, None]], axis=1)
        object_poses = np.repeat(init_poses[None, ...], self.obj_num, axis=0)   # (89, 3, 4)

        for i, obj_idx in enumerate(obj_idxs):
            object_poses[obj_idx, :, :] = poses[:, :, i]
            
        tmp = np.array([0,0,0,1])[None, None, :]
        tmp = np.repeat(tmp, self.obj_num, axis=0)
        object_poses = np.concatenate([object_poses, tmp], axis=1)
        camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))

        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['cloud_segs'] = seg_sampled.astype(np.int64)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses'] = object_poses.astype(np.float32)
        ret_dict['frame_id'] = np.array(self.frameid[index][frame]).astype(np.int64)
        ret_dict['camera_pose'] = camera_poses[self.frameid[index][frame]].astype(np.float32)
        ret_dict['scene_name'] = np.array(int(self.scenename[index].split('_')[1])).astype(np.int64)

        return ret_dict


def load_grasp_labels(root):
    obj_names = list(range(88))
    valid_obj_idxs = []
    grasp_labels = {}
    for i, obj_name in enumerate(tqdm(obj_names, desc='Loading grasping labels...')):
        if i == 18: continue
        valid_obj_idxs.append(i + 1) #here align with label png
    return valid_obj_idxs, grasp_labels


def total_collate_fn(list_data):
    data_list = []
    for i in range(len(list_data[0])):
        frame = [list_data[j]['frame_{}'.format(str(i))] for j in range(len(list_data))]
        data_list.append(collate_fn(frame))

    return data_list


def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key:collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


if __name__ == "__main__":
    TRAIN_DATASET = GraspTracking_Dataset('/data/graspnet')
    TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=2, shuffle=True, drop_last=True, 
                                num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=total_collate_fn)
    for batch_idx, batch_data in enumerate(TRAIN_DATALOADER):
        pass
