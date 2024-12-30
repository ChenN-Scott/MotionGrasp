import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from models.motion_encoder import TemporalEncoder, StateEncoder, SpatialEncoder


class FuseNet(nn.Module):
    def __init__(self, feature_dim=128):
        super(FuseNet, self).__init__()
        self.feature_dim = feature_dim
        self.d_fc1 = nn.Linear(self.feature_dim*4, self.feature_dim*2, bias=True)
        self.d_fc2 = nn.Linear(self.feature_dim*2, self.feature_dim, bias=True)

    def forward(self, pre_feat, cur_feat):
        '''feature fuse for motion_feat, grasp_feat and crop_feat
        cur_feat: (B, M, 2C)
        pre_feat: (B, M, 2C)
        ''' 
        grasp_feat = torch.cat([pre_feat, cur_feat], dim=-1)

        grasp_feat = F.relu(self.d_fc1(grasp_feat), inplace=True)
        grasp_feat = self.d_fc2(grasp_feat)

        return grasp_feat


class MatchNet(nn.Module):
    def __init__(self, feature_dim=256, time_steps=5, loc_dim=7):
        super(MatchNet, self).__init__()
        self.feature_dim = feature_dim
        self.fc = nn.Linear(loc_dim, self.feature_dim//2, bias=False)
        self.fc2 = nn.Linear(self.feature_dim//2, self.feature_dim, bias=False)

        self.d_fc = nn.Linear(self.feature_dim*2, self.feature_dim, bias=False)

        self.coord_fc1 = nn.Linear(self.feature_dim, self.feature_dim//2, bias=False)
        self.coord_fc2 = nn.Linear(self.feature_dim//2, self.feature_dim//4, bias=False)
        self.coord_fc3 = nn.Linear(self.feature_dim//4, 1, bias=False)

        self.coord_fc4 = nn.Linear(self.feature_dim//2, 1, bias=False)

        self.temporal = TemporalEncoder(historical_steps=time_steps, embed_dim=self.feature_dim)
        self.tpe = nn.Embedding(100, self.feature_dim)

        self.rel_emb = StateEncoder(loc_dim, self.feature_dim)
        self.spatial = SpatialEncoder(self.feature_dim)
        self.fuse = FuseNet(self.feature_dim)

    def forward(self, pre_coord_all, cur_coord_all, fr_diff, grasp_feat, crop_feat, grasp_features, crop_features, key_mask=None, pre_mask=None, his_coord_all=None, his_mask=None):
        """
        pre_coord_all: (B, M, L+1, D)
        cur_coord_all: (B, N, D)
        fr_diff: (B, )
        """
        bs, ns, l, d = pre_coord_all.shape
        l = l-1
        if type(pre_mask) == type(None):
            pre_mask = torch.ones(pre_coord_all.shape[0], pre_coord_all.shape[1]).cuda()

        cur_coord_all = cur_coord_all.unsqueeze(1) # (B, 1, N, D), current grasp parameters
        pre_num = pre_coord_all.shape[1]
        cur_num = cur_coord_all.shape[2]

        # compute difference between current grasp and latest grasp
        pre_last = pre_coord_all[..., -1, :3].unsqueeze(2) # (B, M, 1, 3), latest grasp xyz
        cur = cur_coord_all[..., :3].clone() # (B, 1, N, 3), current grasp xyz
        cur_trans = cur - pre_last # (B, M, N, 3), grasp xyz difference between current and latest

        pre_last = pre_coord_all[..., -1, 3:12] # (B, M, 9), latest grasp rot of each tracklet
        cur_rot = cur_coord_all[..., 3:12].squeeze(1).clone() # (B, N, 9), current grasp rot
        cur_rot_diff = compute_rotation_diff(cur_rot, pre_last) # (B, M, N, 9), grasp rot difference between current and latest
        
        cur_coord_all = torch.cat([cur_trans, cur_rot_diff], dim=3) # (B, M, N, 12), current grasp motion vector
        cur = cur_coord_all # (B, M, N, 12)

        # compute cur grasp pose diff
        pre_last = pre_coord_all[..., -1, :]
        pos_emb = self.rel_emb(pre_last) # (B, M, C)

        # compute cur grasp feature diff
        pre_last_grasp_feat = grasp_feat[..., -1, :] # (B, M, 2C)
        pre_last_crop_feat = crop_feat[..., -1, :] # (B, M, 2C)
        pre_last_feat = self.fuse(pre_last_grasp_feat, pre_last_crop_feat).unsqueeze(2)
        cur_feat = self.fuse(grasp_features, crop_features).unsqueeze(1)
        cur_feat_diff = cur_feat - pre_last_feat
        cur_coord_diff = self.fc2(F.relu((self.fc(cur)))) # (B, M, N, C)
        cur_motion_diff = self.d_fc(torch.cat([cur_coord_diff, cur_feat_diff], dim=-1))

        # compute historical grasp motion difference vector
        pre_interval = pre_coord_all[..., 1:, :3] - pre_coord_all[..., :-1, :3] # (B, M, L, 3), history grasp xyz
        pre_rot = compute_rotation_diff(pre_coord_all[..., 1:, 3:12], pre_coord_all[..., :-1, 3:12]) # (B, M, L, 9), history grasp rot difference
        pre = torch.cat([pre_interval, pre_rot], dim=3) # (B, M, L, 12), history grasp motion vector

        pre_grasp_feat_1 = grasp_feat[..., 1:, :]
        pre_crop_feat_1 = crop_feat[..., 1:, :]
        pre_grasp_feat_2 = grasp_feat[..., :-1, :]
        pre_crop_feat_2 = crop_feat[..., :-1, :]
        
        pre_feat_1 = self.fuse(pre_grasp_feat_1, pre_crop_feat_1)
        pre_feat_2 = self.fuse(pre_grasp_feat_2, pre_crop_feat_2)
        pre_feat_diff = pre_feat_1 - pre_feat_2
        pre_coord_diff = self.fc2(F.relu(self.fc(pre)))
        pre_motion_diff = self.d_fc(torch.cat([pre_coord_diff, pre_feat_diff], dim=-1))
        pre_motion_diff = pre_motion_diff.reshape(bs*pre_num, l, -1)

        fr_diff = fr_diff[...,1:] # (B, N, L)
        tpe = self.tpe(fr_diff.long()).view(-1, l, self.tpe.weight.shape[-1]) # time encoding
        pre_motion_diff = pre_motion_diff + tpe # (B*M, L, C)

        mask = torch.zeros(bs*ns, l).bool().cuda()
        pre_motion_diff, pre_motion_diff_tra = self.temporal(pre_motion_diff.permute(1,0,2), mask, key_mask) # (B*M, 1, C)

        pre_motion_diff = pre_motion_diff.squeeze(-2).view(bs, ns, -1) # (B, M, C)
        pre_motion_diff = self.spatial(pre_motion_diff, pre_motion_diff, pos_emb) # (B, M, C) , pre_mask, pre_mask
        pre_feat = pre_motion_diff.clone()

        # compute tracker-grasp pair correspondence scores
        pre_motion_diff = pre_motion_diff.unsqueeze(-2).expand(-1,-1,cur_num,-1)
        motion_diff = pre_motion_diff - cur_motion_diff
        scores = self.coord_fc1(motion_diff) # (B, M, N, C/2)
        scores_fine = self.coord_fc2(F.relu(scores)) # (B, M, N, C/4)
        scores_fine = self.coord_fc3(F.relu(scores_fine)).squeeze(-1) # (B, M, N)

        scores_coarse = self.coord_fc4(F.relu(scores)).squeeze(-1) # (B, M, N, C/2)

        return scores_coarse, scores_fine, pre_feat


def compute_rotation_diff(grasp_1, grasp_2):
    """
    grasp: (B, M/N, 9) or (B, M, L, 9)
    """
    if len(grasp_1.shape) == 3:
        b, m, _ = grasp_1.shape
        _, n, _ = grasp_2.shape
        grasp_rot_1 = grasp_1.reshape(b, m, 3, 3).unsqueeze(2) # (b, m, 1, 3, 3)
        grasp_rot_2 = grasp_2.reshape(b, n, 3, 3).unsqueeze(1) # (b, 1, n, 3, 3)

        rot = torch.matmul(grasp_rot_2.transpose(3,4), grasp_rot_1) # ()
        # rot = torch.diagonal(rot, dim1=3, dim2=4).sum(dim=3) # (b, m, n)
        rot = rot.view(b, m, n, 9)
    else:
        b, m, l, _ = grasp_1.shape
        grasp_rot_1 = grasp_1.reshape(b, m, l, 3, 3)
        grasp_rot_2 = grasp_2.reshape(b, m, l, 3, 3)

        rot = torch.matmul(grasp_rot_2.transpose(3,4), grasp_rot_1)
        rot = rot.view(b, m, l, 9)

    return rot

