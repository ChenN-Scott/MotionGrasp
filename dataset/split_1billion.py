import os
import sys
import numpy as np
import scipy.io as scio
import argparse
import json
import yaml 
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '../utils'))
sys.path.append(os.path.join(ROOT_DIR, 'logs'))
from info_object import obj_names

SEQUENCE_SIZE = 7
PREFIX_POSE = 'registered_'

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/data/GraspNet_1billion', type=str)
parser.add_argument('--billion_dir', default='/data/one_billion', type=str)
cfgs = parser.parse_args()

scene_root = os.path.join(cfgs.data_dir, 'scenes')
billion_root = os.path.join(cfgs.billion_dir, 'scenes')
model_root = os.path.join(cfgs.data_dir, 'models')


def make_dir(root):
    if not os.path.exists(root):
        os.makedirs(root)


def split():
    camera_intrinsics_path = os.path.join(cfgs.billion_dir, 'cam_intrinsics')
    make_dir(camera_intrinsics_path)
    scene_num = len(os.listdir(scene_root))
    for i in tqdm(range(100, scene_num), desc='Processing scene'):
        scene_name = os.path.join(scene_root, 'scene_{}'.format(str(i).zfill(4)))
        realsense_path = os.path.join(scene_name, 'realsense')
        rgb_path = os.path.join(realsense_path, 'rgb')
        depth_path = os.path.join(realsense_path, 'depth')
        meta_path = os.path.join(realsense_path, 'meta')
        
        scene_path = os.path.join(billion_root, 'scene_{}'.format(str(i).zfill(4)))

        image_num = len(os.listdir(rgb_path))
        for camera_sn in range(int((image_num-1)/15)):
            camera_path = os.path.join(scene_path, 'camera_{}_{}'.format(str(i).zfill(4), str(camera_sn).zfill(4)))
            camera_intrinsics = os.path.join(camera_intrinsics_path, 'camera_{}_{}.npy'.format(str(i).zfill(4), str(camera_sn).zfill(4)))
            make_dir(camera_path)

            rgb_new_path = os.path.join(camera_path, 'color')
            depth_new_path = os.path.join(camera_path, 'depth')
            pose_new_path = os.path.join(camera_path, 'pose')
            query_path = os.path.join(camera_path, 'query_grasp_poses')

            make_dir(rgb_new_path)
            make_dir(depth_new_path)
            make_dir(pose_new_path)
            make_dir(query_path)

            for image in range(SEQUENCE_SIZE):
                # save rgb
                rgb_old = os.path.join(rgb_path, '{}.png'.format(str(int(1+camera_sn*15+image)).zfill(4)))
                rgb_new = os.path.join(rgb_new_path, '{}.png'.format(str(image).zfill(4)))
                if not os.path.exists(rgb_new):
                    os.system('cp {} {}'.format(rgb_old, rgb_new))

                # save depth
                depth_old = os.path.join(depth_path, '{}.png'.format(str(int(1+camera_sn*15+image)).zfill(4)))
                depth_new = os.path.join(depth_new_path, '{}.png'.format(str(image).zfill(4)))
                if not os.path.exists(depth_new):
                    os.system('cp {} {}'.format(depth_old, depth_new))

                # save intrinsics
                meta = scio.loadmat(os.path.join(meta_path, '{}.mat'.format(str(int(1+camera_sn*15+image)).zfill(4))))
                camera_intrinsics = os.path.join(camera_intrinsics_path, 'camera_{}_{}.npy'.format(str(i).zfill(4), str(camera_sn).zfill(4)))
                if not os.path.exists(camera_intrinsics):
                    np.save('{}'.format(camera_intrinsics), meta['intrinsic_matrix'])

                # save poses
                obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
                poses = meta['poses']
                pose_new_path_ = os.path.join(pose_new_path, '{}'.format(str(image).zfill(4)))
                make_dir(pose_new_path_)
                for id, obj_id in enumerate(obj_idxs):
                    pose_new = os.path.join(pose_new_path_, PREFIX_POSE+'{}.npy'.format(str(obj_id-1)))
                    pose = poses[:, :, id]
                    np.save('{}'.format(pose_new), pose)


def make_frame_dicts(type):
    frame_dicts = {}

    scene_list = sorted(os.listdir(os.path.join(billion_root)))
    all_type = {'all':scene_list[0:90], 'seen':scene_list[0:30], 'similar':scene_list[30:60], 'novel':scene_list[60:90]}
    offset = {'all':0, 'seen':0, 'similar':30, 'novel':60}
    scene_list_type = all_type[type]
    for id, scene in enumerate(scene_list_type):
        realsense_path = os.path.join(scene_root, scene, 'realsense')
        rgb_path = os.path.join(realsense_path, 'rgb')
        image_num = len(os.listdir(rgb_path))
        meta_path = os.path.join(realsense_path, 'meta')
        scene_dict = {}

        for camera in range(int((image_num-1)/15)):
            camera_sn = '{}_{}'.format(str(id+100+int(offset[type])).zfill(4), str(camera).zfill(4))
            meta = scio.loadmat(os.path.join(meta_path, '0000.mat'))
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            camera_dict = {}
            
            for obj_id in obj_idxs:
                camera_dict[str(obj_id)] = 'to be finished'
            scene_dict[camera_sn] = camera_dict
        
        frame_dicts[scene] = scene_dict
    frame_path = os.path.join(cfgs.billion_dir, 'collision_label_initial_frame', '{}_frame_dicts.json'.format(type))
    make_dir(os.path.join(cfgs.billion_dir, 'collision_label_initial_frame'))

    with open(frame_path, 'w') as f:
        data = json.dumps(frame_dicts)
        f.write(data)
    f.close()


def make_object_name_list():
    object_name_list = []
    save_dict = {}
    object_name_list_path = os.path.join(cfgs.billion_dir, 'default.yaml')

    model_list = sorted(os.listdir(model_root))
    for id, model in enumerate(model_list):
        object_name_list.append('{}.ply'.format(obj_names[id]))
    
    save_dict['object_model_list'] = object_name_list
    with open(object_name_list_path, 'w') as f:
        yaml.dump(data=save_dict, stream=f, allow_unicode=True)
    f.close()


if __name__ == '__main__':
    split()
    make_frame_dicts('all')
    make_frame_dicts('seen')
    make_frame_dicts('similar')
    make_frame_dicts('novel')
    make_object_name_list()