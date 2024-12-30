import os
import sys
import numpy as np
from tqdm import tqdm
import argparse
import time
import json
import copy
import math
import random
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval

import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'eval'))

from models.graspnet import GraspNet
from models.motion_encoder import pred_decode
from models.tracker import merge_frame_end_points, MotionTracker, return_gt_grasp
from dataset.grasptracking_dataset import GraspTracking_Dataset, total_collate_fn
from utils.collision_detector import ModelFreeCollisionDetector
from utils.eval_utils import eval_first_scene
from eval.motiongraspeval import MotionGraspEval
import pointnet2_utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/data/GraspNet_1billion', required=True)
parser.add_argument('--evaluate_root', default=None)
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--model_name', default='pointnet')
parser.add_argument('--checkpoint_path_1', help='GraspNet Model checkpoint path', default="../logs/log_rs/checkpoint-rs.tar")
parser.add_argument('--checkpoint_path_2', help='Grasp Tracker Model checkpoint path', default=None)
parser.add_argument('--log_dir', default='logs/log_train')
parser.add_argument('--split', default='test')
parser.add_argument('--test_type', default='seen')
parser.add_argument('--dump_dir', help='Model checkpoint path', default='preds/')
parser.add_argument('--gt_dir', default=None)
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--frame_size', type=int, default=5, help='Frame Size during testing [default: 5]')
parser.add_argument('--sequence_size', type=int, default=15, help="Sequence Size during testing [default: 15]")
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during testing [default: 2]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--feature_dim', type=int, default=256)
parser.add_argument('--grasp_num', type=int, default=1024)
parser.add_argument('--make', action='store_true', default=False)
parser.add_argument('--infer', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--error', action='store_true', default=False)
parser.add_argument('--rot_type', type=str, default='6d')
cfgs = parser.parse_args()

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler('log.txt')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


# ------------------------------------GLOBAL CONFIG------------------------------------- 
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


@torch.no_grad()
def make_query_grasp():
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test')
    test_dataset = GraspTracking_Dataset(cfgs.dataset_root, valid_obj_idxs=None, grasp_labels=None, 
                                    camera=cfgs.camera, split=cfgs.split, num_points=cfgs.num_point, 
                                    sequence_size=cfgs.sequence_size, remove_outlier=True, augment=False,load_label=False)
    print('Test dataset length: ', len(test_dataset))
    scene_list = test_dataset.scene_list()
    sequence_list = test_dataset.sequence_list()
    test_dataloader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False,
                                 num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=total_collate_fn)
    print('Test dataloader length: ', len(test_dataloader))

    # Init the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net1 = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                        cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04]).to(device)

    # Load checkpoint
    checkpoint_1 = torch.load(os.path.join(cfgs.checkpoint_path_1))

    net1.load_state_dict(checkpoint_1['model_state_dict'])
    start_epoch = checkpoint_1['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path_1, start_epoch))

    net1.eval()
    for batch_idx, batch_data in enumerate(tqdm(test_dataloader)):
        for i in batch_data:
            for key in i.keys():
                i[key] = i[key].to(device)

        # get referenced grasps
        end_points_1 = net1(batch_data[0])

        if 'batch_grasp_preds' in end_points_1.keys():
            grasp_preds = end_points_1['batch_grasp_preds']
        else:
            grasp_preds, _ = pred_decode(end_points_1, remove_background=False) # (B, Ns, 17)
        
        batch_seed_segs = end_points_1['cloud_segs'].cpu()
        batch_seed_inds = end_points_1['fp2_inds'].cpu()

        batch_seed_segs = batch_seed_segs.unsqueeze(1).contiguous().to(torch.float).to(device)
        batch_seed_inds = batch_seed_inds.to(device)
        batch_grasp_segs = pointnet2_utils.gather_operation(batch_seed_segs, batch_seed_inds).squeeze(1).contiguous().to(torch.int)

        for b in range(len(grasp_preds)):
            seed_segs = batch_grasp_segs[b].cpu()

            grasp_indices_ = np.array([i for i in range(grasp_preds[b].shape[0])])
            data_idx = batch_idx * cfgs.batch_size + b
            scene_id = int(end_points_1['scene_name'][b].cpu())
            frame_id = int(end_points_1['frame_id'][b].cpu())
            grasp_group = GraspGroup(grasp_preds[b].cpu().numpy())
            query_dir = os.path.join(cfgs.evaluate_root, 'scenes', scene_list[data_idx],
                                    'camera_{}_{}'.format(str(scene_list[data_idx].split('_')[1]),
                                    str(sequence_list[data_idx]).zfill(4)), 'query_grasp_poses')
            if not os.path.exists(query_dir):
                os.makedirs(query_dir)
            query_json_path = os.path.join(query_dir, 'grasp_query.json')
            query_pose_path = os.path.join(query_dir, 'grasp_group.npy')
            query_indices_path = os.path.join(query_dir, 'grasp_indices.npy')

            # filter grasp with grasp segmentation label
            segs_mask = seed_segs > 0
            grasp_group = grasp_group[segs_mask.cpu().numpy()]
            grasp_group_ = copy.deepcopy(grasp_group)
            grasp_indices_ = grasp_indices_[segs_mask.cpu().numpy()]

            # filter grasp with collision detector
            if cfgs.collision_thresh > 0:
                cloud = test_dataset.get_data(data_idx, 0, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
                collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                grasp_group = grasp_group[~collision_mask]     

            grasp_list, score_list, collision_mask_list = eval_first_scene(ge, scene_id ,frame_id, grasp_group)
            grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(score_list), np.concatenate(collision_mask_list)
            
            # get top10 filtered grasp
            valid_grasp = []
            valid_score = []
            for grasp_pre_obj, score_pre_obj in zip(grasp_list, score_list):
                for grasp, score in zip(grasp_pre_obj, score_pre_obj):
                    if score > 0:
                        valid_grasp.append(grasp)
                        valid_score.append(score)

            if len(valid_grasp) == 0:
                logger.warning('scene:{} sequence: {} do not generate any valid grasp, using lower score bound to select again'.format(
                    str(scene_list[data_idx].split('_')[1]), str(sequence_list[data_idx]).zfill(4)))
                for grasp_pre_obj, score_pre_obj in zip(grasp_list, score_list):
                    for grasp, score in zip(grasp_pre_obj, score_pre_obj):
                        valid_grasp.append(grasp)
                        valid_score.append(score)

            valid_grasp = np.vstack(valid_grasp)
            valid_score = np.vstack(valid_score)
            indices = np.argsort(-valid_score, axis=0)
            if indices.shape[0] < 10:
                first_indices = np.array([indices[0]]*(10-indices.shape[0]))
                indices = np.concatenate([indices, first_indices])

            grasp_list = GraspGroup(valid_grasp[indices[:10]].squeeze(1))
            grasp_indices = grasp_equal(grasp_list, grasp_group_, grasp_indices_)
            grasp_segs = seed_segs[grasp_indices]
            
            # save query grasp files
            grasp_list.save_npy(query_pose_path)
            print('camera_{}_{}'.format(str(scene_list[data_idx].split('_')[1]),
                                    str(sequence_list[data_idx]).zfill(4)), grasp_indices)
            np.save(query_indices_path, grasp_indices)

            query_pose_list = []
            for grasp in range(grasp_list.grasp_group_array.shape[0]):
                grasp_dict = {}
                grasp_dict['start_frame'] = '0000'
                grasp_dict['object_id'] = int(grasp_segs[grasp]-1)
                query_pose_list.append(grasp_dict)
            query_json = json.dumps(query_pose_list)
            with open(query_json_path, 'w') as f:
                f.write(query_json)
                f.close


@torch.no_grad()
def inference():
    # Prepare dataset and dataloader
    test_dataset = GraspTracking_Dataset(cfgs.dataset_root, valid_obj_idxs=None, grasp_labels=None, 
                                    camera=cfgs.camera, split=cfgs.split, num_points=cfgs.num_point, 
                                    sequence_size=cfgs.sequence_size, remove_outlier=True, augment=False,load_label=False)
    print('Test dataset length: ', len(test_dataset))
    
    scene_list = test_dataset.scene_list()
    sequence_list = test_dataset.sequence_list()
    test_dataloader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False,
                                 num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=total_collate_fn)
    print('Test dataloader length: ', len(test_dataloader))

    # Init the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    graspnet = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                        cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04]).to(device)
    tracker = MotionTracker(device=0, track_num=cfgs.grasp_num, feature_dim=cfgs.feature_dim, is_training=False, nfr=cfgs.frame_size).to(device)

    # Load checkpoint
    checkpoint_1 = torch.load(os.path.join(cfgs.checkpoint_path_1))
    checkpoint_2 = torch.load(os.path.join(cfgs.log_dir, cfgs.checkpoint_path_2))
    new_state_dict_1 = OrderedDict()
    for k, v in checkpoint_2['model_state_dict'].items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict_1[name] = v
    graspnet.load_state_dict(checkpoint_1['model_state_dict'])
    tracker.load_state_dict(new_state_dict_1)
    
    start_epoch = checkpoint_1['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path_1, start_epoch))
    print('test: {}'.format(cfgs.log_dir))

    # Predict grasp per batch
    graspnet.eval()
    tracker.eval()
    tic = time.time()
    for batch_idx, batch_data in enumerate(tqdm(test_dataloader)):
        data_idx = batch_idx * cfgs.batch_size
        scene_id = int(scene_list[data_idx].split('_')[1])
        # if scene_id == 187: continue

        for i in batch_data:
            for key in i.keys():
                i[key] = i[key].to(device)

        # get referenced grasps
        end_points = graspnet(batch_data[0])
        if 'batch_grasp_preds' in end_points.keys():
            grasp_preds = end_points['batch_grasp_preds']
        else:
            grasp_preds, _ = pred_decode(end_points, remove_background=False) # (B, Ns, 17)

        # load referenced grasp indices
        grasp_indices_list = []
        query_grasp_list = []
        for b in range(len(grasp_preds)):
            data_idx = batch_idx * cfgs.batch_size + b
            query_dir = os.path.join(cfgs.evaluate_root, 'scenes', scene_list[data_idx],
                                    'camera_{}_{}'.format(str(scene_list[data_idx].split('_')[1]),
                                    str(sequence_list[data_idx]).zfill(4)), 'query_grasp_poses')
            query_indices_path = os.path.join(query_dir, 'grasp_indices.npy')
            query_grasp_path = os.path.join(query_dir, 'grasp_group.npy')
            query_grasp = torch.tensor(np.load(query_grasp_path))
            query_grasp_list.append(query_grasp)
            grasp_indices = torch.tensor(np.load(query_indices_path))
            grasp_indices_list.append(grasp_indices)
        grasp_indices = torch.stack(grasp_indices_list, dim=0)
        query_grasp = torch.stack(query_grasp_list, dim=0)
        batch_grasp_list = []

        for frame_id in range(0, cfgs.sequence_size):
            end_points = graspnet(batch_data[frame_id])
            if 'batch_grasp_preds' in end_points.keys():
                grasp_preds = end_points['batch_grasp_preds']
            else:
                grasp_preds, _ = pred_decode(end_points, remove_background=False) # (B, Ns, 17)

            if frame_id == 0:
                end_points_1 = copy.deepcopy(end_points)
                tracker(frame_id, grasp_preds, end_points_1)
                grasp_preds_1 = pointnet2_utils.gather_operation(grasp_preds.permute(0,2,1).contiguous(), grasp_indices.cuda().to(torch.int)).permute(0,2,1)
                batch_grasp_list.append(grasp_preds_1)
                continue
            else:
                corr_pred, _ = tracker(frame_id, grasp_preds, end_points)

            _, top_indices = torch.max(corr_pred, dim=2)
                
            grasp_preds = pointnet2_utils.gather_operation(grasp_preds.permute(0,2,1).contiguous(), top_indices.cuda().to(torch.int)).permute(0,2,1)
            batch_grasp_list.append(grasp_preds)
        
        new_grasp = tracker.return_memo()
        for frame_id in range(0, cfgs.sequence_size-1):
            grasp_preds = batch_grasp_list[frame_id+1]
            # grasp_preds[..., 13:16] = new_grasp[frame_id][..., :3]
            # grasp_preds[..., 4:13] = new_grasp[frame_id][..., 3:12]
            grasp_preds[..., 13:16] = new_grasp[..., frame_id+1, :3]
            grasp_preds[..., 4:13] = new_grasp[..., frame_id+1, 3:12]
            grasp_preds = pointnet2_utils.gather_operation(grasp_preds.permute(0,2,1).contiguous(), grasp_indices.cuda().to(torch.int)).permute(0,2,1)
            batch_grasp_list[frame_id+1] = grasp_preds
        tracker.clear()

        # save pred grasp files
        for frame in range(cfgs.sequence_size):
            grasp_preds = batch_grasp_list[frame].contiguous()
            for i in range(cfgs.batch_size):
                data_idx = batch_idx * cfgs.batch_size + i
        
                save_dir = os.path.join(cfgs.dump_dir, scene_list[data_idx], 
                                        'camera_{}_{}'.format(str(scene_list[data_idx].split('_')[1]), 
                                        str(sequence_list[data_idx]).zfill(4)), '{}'.format(str(frame).zfill(4)))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                grasp_group = GraspGroup(np.array(grasp_preds[i].cpu()))
                for j in range(len(grasp_group)):
                    save_path = os.path.join(save_dir, '{}.npy'.format(str(j)))
                    np.save(save_path, grasp_group[j].grasp_array, allow_pickle=True)
                    # grasp_group[j].save_npy(save_path)


def compute_grasp_correspondence(grasp_preds, grasp_preds_gt, source_pose, cur_pose):
    grasp_preds_gt = return_gt_grasp(grasp_preds_gt, source_pose, cur_pose)

    bs = grasp_preds.shape[0]
    if grasp_preds.shape[2] > 12:
        grasp_preds = torch.cat([grasp_preds[:, :, 13:16], grasp_preds[:, :, 4:13]], dim=-1)
    if grasp_preds_gt.shape[2] > 12:
        grasp_preds_gt = torch.cat([grasp_preds_gt[:, :, 13:16], grasp_preds_gt[:, :, 4:13]], dim=-1)

    grasp_trans = grasp_preds[:, :, :3]
    grasp_rot = grasp_preds[:, :, 3:12].reshape(bs, grasp_preds.shape[1], 3, 3)
    gt_trans = grasp_preds_gt[:, :, :3]
    gt_rot = grasp_preds_gt[:, :, 3:12].reshape(bs, grasp_preds.shape[1], 3, 3)

    # compute translation correspondence 
    trans_correspondence = torch.norm(gt_trans - grasp_trans, dim=2)
    trans_valid_mask = trans_correspondence <= 0.1
    
    # compute rotation correspondence
    rot_correspondence = torch.matmul(gt_rot, grasp_rot.transpose(2,3))
    rot_correspondence = torch.diagonal(rot_correspondence, dim1=2, dim2=3).sum(dim=2)
    rot_correspondence = torch.clamp(rot_correspondence, 1, 3)
    rot_correspondence = (rot_correspondence-1)/2
    rot_correspondence = torch.acos(rot_correspondence)
    # rot_correspondence = 0.4 * (rot_correspondence / np.pi)
    rot_valid_mask = rot_correspondence <= 30.0 / 180.0 * np.pi

    # compute valid grasp correspondence
    valid_mask = trans_valid_mask & rot_valid_mask

    # compute overall grasp correspondence
    return valid_mask


def grasp_equal(grasp_group1, grasp_group2, grasp_indices):
    # judge the grasp from group1 equals to which grasp from group2, return indices of grasp_group2
    grasp_indices_ = []
    for grasp_1 in grasp_group1:
        for id, grasp_2 in enumerate(grasp_group2):
            if (grasp_1.translation == grasp_2.translation).all()  \
                and (grasp_1.rotation_matrix == grasp_2.rotation_matrix).all():
                grasp_indices_.append(grasp_indices[id])
                break
    
    return np.array(grasp_indices_)


def evaluate():
    pred_path = cfgs.dump_dir
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    mgev = MotionGraspEval(cfgs.evaluate_root, log_dir=None, pred_dir=pred_path, type=cfgs.test_type)
    print('='*40)
    print('Evaluating scene')

    mgta = mgev.eval_mgta_all(cfgs.gt_dir)
    print("Done.")
    print('=====================')
    print('test: {}, split:{}'.format(cfgs.log_dir, cfgs.test_type))


def compute_correspondence_object_pose(grasp_preds, seed_inds, end_points, end_points_gt, grasp_gt):
    B, Ns, _ = grasp_preds.size()
    grasp_trans_preds, grasp_trans_gt = grasp_preds[:, :, 13:16], grasp_gt[:, :, 13:16]
    grasp_rot_preds, grasp_rot_gt = grasp_preds[:, :, 4:13].view(B, Ns, 3, 3), grasp_gt[:, :, 4:13].view(B, Ns, 3, 3)
    preds_poses = []
    gt_poses = []
    for batch_id in range(B):
        # preds object poses
        preds_cloud_segs = end_points['cloud_segs'][batch_id]
        preds_object_poses = end_points['object_poses'][batch_id]
        preds_seed_inds = seed_inds.long()[batch_id]
        
        preds_seed_segs = preds_cloud_segs[preds_seed_inds]
        preds_seed_poses = preds_object_poses[preds_seed_segs]
        preds_poses.append(preds_seed_poses)
        
        # gt object poses
        gt_cloud_segs = end_points_gt['cloud_segs'][batch_id]
        gt_object_poses = end_points_gt['object_poses'][batch_id]
        gt_seed_inds = end_points_gt['fp2_inds'][batch_id].long()
        
        gt_seed_segs = gt_cloud_segs[gt_seed_inds]
        gt_seed_poses = gt_object_poses[gt_seed_segs]
        gt_poses.append(gt_seed_poses)

    preds_poses = torch.stack(preds_poses, dim=0).contiguous()
    gt_poses = torch.stack(gt_poses, dim=0).contiguous()
    preds_poses_inv = torch.inverse(preds_poses)
    gt_poses_inv = torch.inverse(gt_poses)
    
    # translation correspondence
    grasp_trans_preds = torch.matmul(preds_poses_inv[:, :, :3, :3], grasp_trans_preds.unsqueeze(-1)).squeeze(-1)
    grasp_trans_preds = grasp_trans_preds + preds_poses_inv[:, :, :3, 3]
    grasp_trans_gt = torch.matmul(gt_poses_inv[:, :, :3, :3], grasp_trans_gt.unsqueeze(-1)).squeeze(-1)
    grasp_trans_gt = grasp_trans_gt + gt_poses_inv[:, :, :3, 3]
    trans_correspondence = torch.norm(grasp_trans_preds-grasp_trans_gt, dim=2) <= 0.1
    
    # rotation correspondence
    grasp_rot_preds = torch.matmul(preds_poses_inv[:, :, :3, :3], grasp_rot_preds)
    grasp_rot_gt = torch.matmul(gt_poses_inv[:, :, :3, :3], grasp_rot_gt)
    rot_correspondence = torch.matmul(grasp_rot_preds, grasp_rot_gt.transpose(2,3))
    rot_correspondence = torch.diagonal(rot_correspondence, dim1=2, dim2=3).sum(dim=2)
    rot_correspondence = torch.clamp(rot_correspondence, 1, 3)
    rot_correspondence = torch.acos((rot_correspondence-1)/2) <= 30.0 / 180.0 * np.pi

    valid_mask = trans_correspondence & rot_correspondence
    
    return valid_mask


def compute_trans_and_rot_diff(type):
    pred_path = cfgs.dump_dir
    gt_path = cfgs.gt_dir

    trans = []
    rot = []
    scene_list = sorted(os.listdir(gt_path))
    all_type = {'all':scene_list, 'seen':scene_list[0:30], 'similar':scene_list[30:60], 'novel':scene_list[60:]}
    for scene_id, scene in enumerate(tqdm(all_type[type])):
        # if int(scene.split('_')[-1]) >= 187: continue
        pred_camera_list = os.listdir(os.path.join(pred_path, scene))
        gt_camera_list = sorted(os.listdir(os.path.join(gt_path, scene)))
        trans_scene = []
        rot_scene = []

        for camera_id, camera_sn in enumerate(gt_camera_list):
            pred_frame_list = os.listdir(os.path.join(pred_path, scene, camera_sn))
            gt_frame_list = sorted(os.listdir(os.path.join(gt_path, scene, camera_sn)))
            trans_camera = []
            rot_camera = []

            for frame_id, frame in enumerate(gt_frame_list):
                pred_grasp_list = os.listdir(os.path.join(pred_path, scene, camera_sn, frame))
                gt_grasp_list = sorted(os.listdir(os.path.join(gt_path, scene, camera_sn, frame)))
                trans_frame = []
                rot_frame = []

                for grasp_id, grasp in enumerate(gt_grasp_list):
                    pred_grasp = np.load(os.path.join(pred_path, scene, camera_sn, frame, grasp))
                    gt_grasp = np.load(os.path.join(gt_path, scene, camera_sn, frame, grasp))
                                       
                    trans_diff = np.linalg.norm(pred_grasp[None,:][:,13:16]-gt_grasp[None,:][:,13:16])
                    
                    pred_rot = pred_grasp[4:13].reshape(3,3)
                    gt_rot = gt_grasp[4:13].reshape(3,3)

                    rot_diff = (np.trace(np.dot(np.linalg.inv(gt_rot), pred_rot))-1)/2
                    rot_diff = np.clip(rot_diff, -1, 1)
                    rot_diff = 180*(abs(math.acos(rot_diff)))/math.pi

                    trans_frame.append(trans_diff)
                    rot_frame.append(rot_diff)
                
                trans_camera.append(sum(trans_frame)/len(trans_frame))
                rot_camera.append(sum(rot_frame)/len(rot_frame))

            trans_scene.append(sum(trans_camera)/len(trans_camera))
            rot_scene.append(sum(rot_camera)/len(rot_camera))

        trans.append(sum(trans_scene)/len(trans_scene))
        rot.append(sum(rot_scene)/len(rot_scene))
        # print('trans error: {}, {}'.format(scene, sum(trans_scene)/len(trans_scene)))
        # print('rot error: {}, {}'.format(scene, sum(rot_scene)/len(rot_scene)))

    print('=====================')
    print('test: {}, split:{}'.format(cfgs.log_dir, cfgs.test_type))
    print('average trans error: {}, rot error: {}'.format(sum(trans)/len(trans), sum(rot)/len(rot)))


if __name__ == '__main__':
    # make_query_grasp() 
    if cfgs.make:
        setup_seed(30)
        make_query_grasp()
    if cfgs.infer:
        setup_seed(30)
        inference()
    if cfgs.eval:
        evaluate()
    if cfgs.error:
        compute_trans_and_rot_diff(cfgs.test_type)



