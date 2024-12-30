import torch
import os
import sys
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from models.motion_encoder import StateEncoder, init_weights
from models.motion_matcher import FuseNet
from pytorch3d.transforms import rotation_6d_to_matrix, quaternion_to_matrix

class Grasp_Aligner(nn.Module):
    def __init__(self, feature_dim=128, loc_dim=12, rot='6d', nfr=5):
        super(Grasp_Aligner, self).__init__()
        self.feature_dim = feature_dim
        self.loc_dim = loc_dim
        self.rot = rot
        self.nfr = nfr

        self.grasp_emb = StateEncoder(self.loc_dim, self.feature_dim)
        self.fc = nn.Linear(self.feature_dim*2, self.feature_dim*2, bias=False)
        self.fc2 = nn.Linear(self.feature_dim*2, self.feature_dim, bias=False)

        self.attn_aligner = AttentionNet(self.nfr, self.feature_dim*2)

        self.fuse = FuseNet(self.feature_dim)
        self.fc_1 = nn.Linear(self.feature_dim*2, self.feature_dim*2, bias=False)
        self.fc_2 = nn.Linear(self.feature_dim*5, self.feature_dim*4, bias=False)
        self.fc_3 = nn.Linear(self.feature_dim*4, self.feature_dim*3, bias=False)
        self.fc_4 = nn.Linear(self.feature_dim*3, self.feature_dim*2, bias=False)

        if self.rot == '6d':
            self.to_rot = nn.Sequential(
                        nn.Linear(self.feature_dim, self.feature_dim//2),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.feature_dim//2, 6),
            )
        else:
            self.to_rot = nn.Sequential(
                        nn.Linear(self.feature_dim, self.feature_dim//2),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.feature_dim//2, 4),
            )

        self.to_translation = nn.Sequential(
                    nn.Linear(self.feature_dim, self.feature_dim//2),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.feature_dim//2, 3)
        )

    def forward(self, his_grasp, target_grasp, his_grasp_feat, target_grasp_feat, his_crop_feat, target_crop_feat, pre_motion_feat, key_mask):
        b, n, l, d = his_grasp.shape
        if target_grasp.shape[2] > 12:
            target_grasp = torch.cat([target_grasp[:, :, 13:16], target_grasp[:, :, 4:13]], dim=-1)

        target_coord_feat = self.grasp_emb(target_grasp) # (B, N, C)
        history_coord_feat = self.grasp_emb(his_grasp) # (B, N, L, C)
        target_coord_feat = target_coord_feat.unsqueeze(2).repeat(1,1,self.nfr,1).contiguous() # (B, N, L, C)

        coord_feat = torch.cat([history_coord_feat, target_coord_feat], dim=-1) # (B, N, L, 2C)
        coord_feat = F.relu(self.fc(coord_feat)) # (B, N, L, 2C)

        his_feat = self.fuse(his_grasp_feat, his_crop_feat)
        target_feat = self.fuse(target_grasp_feat, target_crop_feat).unsqueeze(2).repeat(1,1,self.nfr,1).contiguous()

        grasp_feat = torch.cat([his_feat, target_feat], dim=-1)
        grasp_feat = F.relu(self.fc_1(grasp_feat)) # (B, N, L, 2C)

        pre_motion_feat = pre_motion_feat.unsqueeze(2).repeat(1,1,self.nfr,1).contiguous()
        all_feat = self.fc_2(torch.cat([coord_feat, grasp_feat, pre_motion_feat], dim=-1))
        all_feat = self.fc_4(F.relu(self.fc_3(F.relu(all_feat)))) # (B, N, L, 2C)

        all_feat = self.attn_aligner(all_feat.permute(2,0,1,3).reshape(l, b*n, -1), src_key_padding_mask=key_mask).reshape(l, b, n, -1).permute(1,2,0,3) # (B, N, L, 2C)
        all_feat = self.fc2(all_feat) # (B, N, L, C)

        rot_pred = self.to_rot(all_feat)
        trans_pred = self.to_translation(all_feat) # (B, N, L, 3)
        if self.rot == '6d':
            rot_pred = rotation_6d_to_matrix(rot_pred) # (B, N, L, 3, 3)
            trans_pred += his_grasp[..., :3]
            rot_pred = torch.matmul(rot_pred.transpose(3,4), his_grasp[..., 3:12].reshape(b,n,l,3,3))
            return torch.cat([trans_pred, rot_pred.reshape(b, n, l, -1)], dim=-1).detach(), trans_pred, rot_pred
        else:
            rot_pred = F.normalize(rot_pred, dim=-1)
            rot_mat = quaternion_to_matrix(rot_pred)
            return torch.cat([trans_pred, rot_mat.reshape(b, n, l, -1)], dim=-1).detach(), trans_pred, rot_pred


def compute_rotation_diff(grasp_1, grasp_2):
    """
    grasp: (B, N, 9)
    """

    b, n, _ = grasp_1.shape
    grasp_rot_1 = grasp_1.reshape(b, n, 3, 3) # (b, n, 3, 3)
    grasp_rot_2 = grasp_2.reshape(b, n, 3, 3) # (b, n, 3, 3)

    rot = torch.matmul(grasp_rot_2.transpose(2,3), grasp_rot_1) # (b, n, 3, 3)
    rot = rot.view(b, n, 9)

    return rot
        

class AttentionNet(nn.Module):
    def __init__(self, historical_steps, embed_dim: int, num_heads: int=8, num_layers: int=8, dropout: float=0.1):
        super(AttentionNet, self).__init__()
        encoder_layer = AttnEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
        attn_mask = self.generate_square_subsequent_mask(historical_steps)
        self.register_buffer('attn_mask', attn_mask)
        nn.init.normal_(self.pos_embed, mean=0, std=.02)
        self.apply(init_weights)

    def forward(self, x, padding_mask=None, src_key_padding_mask=None):
        # x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x)
        x = x + self.pos_embed
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=src_key_padding_mask)
        return out
    

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int):
        mask = (torch.triu(torch.ones(seq_len, seq_len))==1).transpose(0,1)
        mask = torch.fliplr(mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0)))
        return mask.cuda()


class AttnEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int=8, dropout: float=0.1):
        super(AttnEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim*4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim*4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)