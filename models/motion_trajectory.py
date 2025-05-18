import numpy as np
import torch
import time

class MotionTrajectory(object):
    def __init__(self, device, grasp_preds, cur_frame, source_pose = None, grasp_features=None, crop_features=None, is_training = True):
        self.device = device
        self.feature_dim = 256
        self.grasp_preds = grasp_preds
        self.pos_dim = 3
        self.rot_dim = 9
        self.loc_dim = 12
        self.nfrs = 5
        self.max_nfrs = 7
        self.bs = grasp_preds.shape[0]
        self.num_trajectory = grasp_preds.shape[1]
        self.init_flag = True
        self.grasp_state = self.grasp_preds.cpu().numpy()
        self.source_pose = source_pose
        self.is_training = is_training

        self.ref_history = torch.from_numpy(np.tile(self.grasp_state[..., None, :], 
                                                    (1, 1, self.nfrs+1, 1))).cuda()
        self.grasp_semantic_history = torch.from_numpy(np.tile(grasp_features[..., None, :], 
                                                               (1, 1, self.nfrs+1, 1))).cuda()
        self.crop_semantic_history = torch.from_numpy(np.tile(crop_features[..., None, :], 
                                                              (1, 1, self.nfrs+1, 1))).cuda()
        self.cur_frame = torch.from_numpy(np.tile(cur_frame, (self.bs, self.num_trajectory, 
                                                              self.nfrs+1, 1))).cuda()
        if not self.is_training:
            self.return_ref_history = torch.from_numpy(np.tile(self.grasp_state[..., None, :],
                                                               (1, 1, self.max_nfrs, 1))).cuda()

    @staticmethod
    def compute_rot_diff(origin_rot: torch.Tensor,
                         input_rot: torch.Tensor) -> torch.Tensor:
        if origin_rot.shape[0] == 1:
            origin_rot = origin_rot.reshape(3,3)
            input_rot = input_rot.reshape(3,3)
        rot_mat = torch.matmul(origin_rot.t(), input_rot)
        rot_diff = torch.acos(torch.clamp((torch.diagonal(rot_mat).sum(0)-1)/2, -1, 1))
        return rot_diff
    
    @staticmethod
    def update_array(origin_array: torch.Tensor,
                     input_array: torch.Tensor) -> torch.Tensor:
        new_array = origin_array.clone()
        new_array[..., :-1, :] = origin_array[..., 1:, :]
        new_array[..., -1:, :] = input_array.unsqueeze(2)
        return new_array.contiguous()
    
    @staticmethod
    def update_array_all(origin_array: torch.Tensor,
                     input_array: torch.Tensor,
                     length: torch.Tensor) -> torch.Tensor:
        new_array = origin_array.clone()
        new_array[..., :-1, :] = origin_array[..., :-1, :]
        new_array[..., length:, :] = input_array[..., length:, :]
        return new_array.contiguous()
    
    def _update_history(self, grasp_preds, cur_frame, length, grasp_features=None, crop_features=None):
        self.ref_history = self.update_array_all(self.ref_history, grasp_preds, length)
        self.cur_frame = self.update_array(self.cur_frame, torch.tensor(cur_frame).unsqueeze(0).unsqueeze(0)\
                                           .repeat(self.bs, self.num_trajectory, 1).contiguous().cuda())
        self.grasp_semantic_history = self.update_array(self.grasp_semantic_history, grasp_features)
        self.crop_semantic_history = self.update_array(self.crop_semantic_history, crop_features)

    def update(self, grasp_preds, cur_frame, length, grasp_features, crop_features):
        self.grasp_state = grasp_preds
        self._update_history(grasp_preds, cur_frame, length, grasp_features, crop_features)
        if not self.is_training:
            self.return_ref_history[..., :-1, :] = self.return_ref_history[..., 1:, :]
            self.return_ref_history[..., (-self.nfrs-1):, :] = self.ref_history[...]

    def get_matching_history(self):
        return self.ref_history[..., (-self.nfrs-1):, :].contiguous(), \
                self.grasp_semantic_history[..., (-self.nfrs-1):, :].contiguous(), \
                self.crop_semantic_history[..., (-self.nfrs-1):, :].contiguous()
    
    def get_fr_idx(self):
        return self.cur_frame[..., (-self.nfrs-1):, :].contiguous()
    
    def get_state(self):
        return self.grasp_state.flatten()
    
