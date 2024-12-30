import os
import sys
import numpy as np
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from backbone import Pointnet2Backbone
from modules import ApproachNet, CloudCrop, OperationNet, ToleranceNet
from label_generation import process_grasp_labels, match_grasp_view_and_label
from pointnet2_utils import CylinderQueryAndGroup


class GraspNetStage1(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300):
        super().__init__()
        self.backbone = Pointnet2Backbone(input_feature_dim)
        self.vpmodule = ApproachNet(num_view, 256)

    def forward(self, end_points):
        pointcloud = end_points['point_clouds']
        seed_features, seed_xyz, end_points = self.backbone(pointcloud, end_points)
        end_points = self.vpmodule(seed_xyz, seed_features, end_points)
        return end_points


class GraspNetStage2(nn.Module):
    def __init__(self, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=True):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.is_training = is_training
        self.crop = CloudCrop(64, 3, cylinder_radius, hmin, hmax_list)
        self.operation = OperationNet(num_angle, num_depth)
        self.tolerance = ToleranceNet(num_angle, num_depth)
    
    def forward(self, end_points):
        pointcloud = end_points['input_xyz']
        if self.is_training:
            grasp_top_views_rot, _, _, _, end_points = match_grasp_view_and_label(end_points)
            seed_xyz = end_points['batch_grasp_point']
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']
            seed_xyz = end_points['fp2_xyz']

        vp_features, vp_seed_features = self.crop(seed_xyz, pointcloud, grasp_top_views_rot)
        end_points = self.operation(vp_features, end_points)
        end_points = self.tolerance(vp_features, end_points)
        end_points['crop_features'] = vp_features

        return end_points


class GraspNet(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False):
        super().__init__()
        self.is_training = is_training
        self.view_estimator = GraspNetStage1(input_feature_dim, num_view)
        self.grasp_generator = GraspNetStage2(num_angle, num_depth, cylinder_radius, hmin, hmax_list, is_training)

    def forward(self, end_points):
        end_points = self.view_estimator(end_points)
        if self.is_training:
            end_points = process_grasp_labels(end_points)
        end_points = self.grasp_generator(end_points)
        return end_points


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

