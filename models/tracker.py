import torch
import torch.nn as nn
import numpy as np
import os
import sys
from typing import List
from graspnetAPI.graspnet_eval import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

ROT_THRESHOLD = 0.1
TRANS_THRESHOLD = 0.3

from models.motion_matcher import MatchNet
from models.motion_trajectory import MotionTrajectory
from models.grasp_aligner import Grasp_Aligner
from pointnet2 import pointnet2_utils


class MotionTracker(nn.Module):
    def __init__(self, device, feature_dim=256, nfr=5, track_num=1024, rot_type='6d', is_training=True):
        super(MotionTracker, self).__init__()
        self.feature_dim = feature_dim
        self.nfr = nfr
        self.num_trajectory = track_num
        self.rot_type = rot_type
        self.label = None
        self.first_grasp = None
        self.first_grasp_features = None
        self.first_crop_features = None
        self.device = device
        self.is_training = is_training
        self.loc_dim = 12

        self.trajectories = None
        self.tracker = MatchNet(self.feature_dim, self.nfr, loc_dim=self.loc_dim)
        self.aligner = Grasp_Aligner(self.feature_dim, self.loc_dim, self.rot_type, self.nfr)

    @property
    def empty(self):
        return False if self.trajectories else True
    
    def clear(self):
        self.label = None 
        self.trajectories = None
        if self.first_grasp is not None:
            self.first_grasp = None    
        if self.first_crop_features is not None:
            self.first_crop_features = None
        if self.first_grasp_features is not None:
            self.first_grasp_features = None    

    def return_memo(self):
        if self.trajectories is None:
            return None
        else:
            batch_grasp_list = self.trajectories.return_ref_history
            return batch_grasp_list

    def update_memo(self, grasp_preds, cur_frame, new_grasp=None, grasp_features=None, crop_features=None):

        if new_grasp is not None:
            grasp_gt = new_grasp
        else:
            grasp_gt = grasp_preds
        length = max(-cur_frame, -self.nfr)
        
        if self.trajectories is not None:
            self.trajectories.update(grasp_gt, cur_frame, length, grasp_features.cpu(), crop_features.cpu())
        else:
            self.trajectories = MotionTrajectory(self.device, grasp_gt, cur_frame, \
                                                 grasp_features=grasp_features.cpu(), crop_features=crop_features.cpu(), \
                                                    is_training = self.is_training)

    def forward(self, cur_frame, grasp_preds, end_points):
        """
        compute the grasp-trajectory pair correspondence relation scores
        """
        # save label
        if self.is_training:
            bs = grasp_preds.shape[0]
            pose = []
            training_mask = []
            for b in range(bs):
                segs = end_points['cloud_segs'][b]
                grasp_indices = end_points['fp2_inds'][b].long()
                grasp_segs = segs[grasp_indices]
                training_mask.append(grasp_segs > 0)
                grasp_pose = end_points['camera_pose'][b]
                pose.append(grasp_pose)

            training_mask = torch.stack(training_mask, dim=0)
            pose = torch.stack(pose).to(grasp_preds.device)
            if self.label is None:
                self.label = {}
                mask = grasp_nms(grasp_preds)
                self.label['source_pose'] = pose
                self.label['source_grasp'] = grasp_preds
                self.label['training_mask'] = training_mask & mask
                self.label['grasp_features'] = end_points['fp2_features'].permute(0,2,1).contiguous()
                self.label['crop_features'] = end_points['crop_features'][:, :, :, -1].permute(0,2,1).contiguous()
                self.label['pose'] = []
            self.label['pose'].append(pose)
        else:
            bs = grasp_preds.shape[0]
            training_mask = []
            for b in range(bs):
                segs = end_points['cloud_segs'][b]
                grasp_indices = end_points['fp2_inds'][b].long()
                grasp_segs = segs[grasp_indices]
                training_mask.append(grasp_segs>0)
            training_mask = torch.stack(training_mask, dim=0)
            if self.first_grasp is None:
                self.first_grasp = grasp_preds
            if self.first_grasp_features is None:
                self.first_grasp_features = end_points['fp2_features'].permute(0,2,1).contiguous()
            if self.first_crop_features is None:
                self.first_crop_features = end_points['crop_features'][:, :, :, -1].permute(0,2,1).contiguous()
            
        # init tracker
        bs, ns = grasp_preds.shape[0], grasp_preds.shape[1]
        grasp_features = end_points['fp2_features'].permute(0,2,1).contiguous()
        crop_features = end_points['crop_features'][:, :, :, -1].permute(0,2,1).contiguous()

        if not self.empty:
            tracker_feat, grasp_feat, crop_feat = self.trajectories.get_matching_history()
            cur_fr_list = self.trajectories.get_fr_idx().squeeze(-1)

            fr_diff = cur_frame - cur_fr_list

            key_mask = torch.zeros(tracker_feat.shape[0]*tracker_feat.shape[1], self.nfr+1).bool().cuda()
            if cur_frame < self.nfr:
                key_mask[..., :(self.nfr-cur_frame)] = True

            grasp_preds = torch.cat([grasp_preds[:, :, 13:16], grasp_preds[:, :, 4:13]], dim=-1)
            if self.is_training:
                corr_pred_coarse, corr_pred_fine, pre_motion_feat = self.tracker(tracker_feat, grasp_preds, fr_diff, grasp_feat, crop_feat, \
                                         grasp_features, crop_features, key_mask, self.label['training_mask'])
            else:
                _, corr_pred_fine, pre_motion_feat = self.tracker(tracker_feat, grasp_preds, fr_diff, grasp_feat, crop_feat, \
                                         grasp_features, crop_features, key_mask)

            _, top_indices = torch.max(corr_pred_fine, dim=2)
            grasp_preds = pointnet2_utils.gather_operation(grasp_preds.permute(0,2,1).contiguous(), \
                                                               top_indices.cuda().to(torch.int)).permute(0,2,1)
            cur_grasp_feat = pointnet2_utils.gather_operation(grasp_features.permute(0,2,1).contiguous(), \
                                                                  top_indices.cuda().to(torch.int)).permute(0,2,1)
            cur_crop_feat = pointnet2_utils.gather_operation(crop_features.permute(0,2,1).contiguous(), \
                                                                 top_indices.cuda().to(torch.int)).permute(0,2,1)

            grasp_list = torch.cat([tracker_feat[..., 2:, :], grasp_preds.unsqueeze(2)], dim=2)
            grasp_feat_list = torch.cat([grasp_feat[..., 2:, :], cur_grasp_feat.unsqueeze(2)], dim=2)
            crop_feat_list = torch.cat([crop_feat[..., 2:, :], cur_crop_feat.unsqueeze(2)], dim=2)
            if self.is_training:
                new_grasp, trans, rot = self.aligner(grasp_list, self.label['source_grasp'], grasp_feat_list, self.label['grasp_features'], \
                                                        crop_feat_list, self.label['crop_features'], pre_motion_feat, key_mask[:, :-1])
                self.update_memo(grasp_preds, cur_frame, new_grasp=new_grasp, grasp_features=cur_grasp_feat, crop_features=cur_crop_feat)
                return corr_pred_coarse, corr_pred_fine, self.label['training_mask'], trans, rot, self.label['pose']
            else:
                new_grasp, trans, rot = self.aligner(grasp_list, self.first_grasp, grasp_feat_list, self.first_grasp_features, \
                                                       crop_feat_list, self.first_crop_features, pre_motion_feat, key_mask[:, :-1])
                self.update_memo(grasp_preds, cur_frame, new_grasp=new_grasp, grasp_features=cur_grasp_feat, crop_features=cur_crop_feat)
                return corr_pred_fine, new_grasp[..., -1, :]
        else:
            grasp_preds = torch.cat([grasp_preds[:, :, 13:16], grasp_preds[:, :, 4:13]], dim=-1)        
            self.update_memo(grasp_preds, cur_frame, grasp_features=grasp_features, crop_features=crop_features)
            return training_mask


def return_gt_grasp(source_grasp, source_pose, query_pose):

    if source_grasp.shape[-1] > 12:
        source_grasp = torch.cat([source_grasp[:, :, 13:16], source_grasp[:, :, 4:13]], dim=-1) 

    source_pose = source_pose.unsqueeze(1).repeat(1, source_grasp.shape[1],1,1)
    query_pose_inv = torch.inverse(query_pose).unsqueeze(1).repeat(1,source_grasp.shape[1],1,1) 

    source_grasp_mat = torch.zeros_like(source_pose)
    source_grasp_mat[:, :, 3, 3] = 1
    source_grasp_trans = source_grasp[:, :, 0:3]
    source_grasp_rot = source_grasp[:, :, 3:12].view(source_grasp.shape[0], source_grasp.shape[1], 3, 3)

    source_grasp_mat[:, :, :3, 3] = source_grasp_trans
    source_grasp_mat[:, :, :3, :3] = source_grasp_rot

    grasp_gt = torch.matmul(query_pose_inv, source_pose)
    grasp_gt = torch.matmul(grasp_gt, source_grasp_mat)

    return torch.cat([grasp_gt[:, :, :3, 3].squeeze(-1), \
                      grasp_gt[:, :, :3, :3].reshape(source_grasp.shape[0], source_grasp.shape[1], -1)], dim=-1).to(source_grasp.device)


@torch.no_grad()
def merge_frame_end_points(dict_list: List[dict]):
    length = len(dict_list)
    end_points = {}
    for key in dict_list[0].keys():
        if dict_list[0][key] == None:
            end_points[key] = None
        else:
            data_list = []
            for i in range(dict_list[0][key].shape[0]):   # for each batch
                for j in range(length):   # for each end_point
                    data_list.append(dict_list[j][key][i, ...].cuda().contiguous())
            end_points[key] = torch.stack(data_list).cuda().contiguous()

    # save loss information to each end_point
    for key in dict_list[-1].keys():
        if 'loss' in key:
            end_points[key] = dict_list[-1][key]
    return end_points


def grasp_nms(grasp_preds):
    bs, ns, _ = grasp_preds.shape
    nms_mask = []
    for b in range(bs):
        grasp_old = GraspGroup(grasp_preds[b].cpu().numpy())
        grasp_new = grasp_old.nms(0.03, 30.0/180 * np.pi)
        grasp_mask = np.zeros([ns], dtype=bool)
        
        for grasp_1 in grasp_new:
            for id, grasp_2 in enumerate(grasp_old):
                if (grasp_1.translation == grasp_2.translation).all() \
                    and (grasp_1.rotation_matrix == grasp_2.rotation_matrix).all():
                    grasp_mask[id] = True
                    break
        
        nms_mask.append(torch.tensor(grasp_mask))

    nms_mask = torch.stack(nms_mask, dim=0)
    return nms_mask.cuda()
