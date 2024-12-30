import os
import sys
import numpy as np
from datetime import datetime
import argparse
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.distributed as dist
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from models.graspnet import GraspNet
from models.motion_encoder import pred_decode
from models.tracker import MotionTracker, merge_frame_end_points
from models.loss import get_loss
from dataset.grasptracking_dataset import GraspTracking_Dataset, load_grasp_labels, total_collate_fn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True)
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--checkpoint_path_1', help='GraspNet Model checkpoint path', default=None)
parser.add_argument('--checkpoint_path_2', help='Grasp Tracker Model checkpoint path', default=None)
parser.add_argument('--log_dir', default='logs/log_ex')
parser.add_argument('--gpu_num', type=int, default=8, help='Number of gpu to use')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 18]')
parser.add_argument('--model_name', type=str, default='pointnet')
parser.add_argument('--frame_size', type=int, default=5, help='Frame Size during training [default: 5]')
parser.add_argument('--sequence_size', type=int, default=7, help="Sequence Size during training [default: 7]")
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.005, help='Initial learning rate [default: 0.001]')
parser.add_argument('--grasp_num', type=int, default=1024)
parser.add_argument('--feature_dim', type=int, default=256)
parser.add_argument('--rot_type', type=str, default='6d')
parser.add_argument('--local_rank', type=int, default=-1)

cfgs = parser.parse_args()

# ------------------------------------GLOBAL CONFIG------------------------------------- 
EPOCH_CNT = 0
DEFAULT_CHECKPOINT_PATH = os.path.join(cfgs.log_dir, 'e17_checkpoint.tar')
CHECKPOINT_PATH_1 = cfgs.checkpoint_path_1 if cfgs.checkpoint_path_1 is not None \
    else DEFAULT_CHECKPOINT_PATH
CHECKPOINT_PATH_2 = cfgs.checkpoint_path_2
if cfgs.local_rank == 0 and not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

valid_obj_idxs, grasp_labels = load_grasp_labels(cfgs.dataset_root)
torch.cuda.set_device(cfgs.local_rank)
dist.init_process_group(backend="nccl")
TRAIN_DATASET = GraspTracking_Dataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, 
                                camera=cfgs.camera, split='train', num_points=cfgs.num_point, 
                                sequence_size=cfgs.sequence_size, remove_outlier=True, augment=True)
TRAIN_SAMPLER = torch.utils.data.distributed.DistributedSampler(TRAIN_DATASET, shuffle=True)
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, sampler=TRAIN_SAMPLER, drop_last=True, 
                              num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=total_collate_fn)
if cfgs.local_rank == 0:
    print('train dataset length: ', len(TRAIN_DATASET))
    print('train dataloader length: ', len(TRAIN_DATALOADER))

# Init models
tracker = MotionTracker(device=cfgs.local_rank, track_num=cfgs.grasp_num, feature_dim=cfgs.feature_dim, rot_type=cfgs.rot_type, nfr=5).cuda()
if cfgs.local_rank == 0:
    print("motiontrakcer init")
tracker = torch.nn.parallel.DistributedDataParallel(tracker, device_ids=[cfgs.local_rank], broadcast_buffers=False,
                                                      output_device=cfgs.local_rank, find_unused_parameters=False)
if cfgs.local_rank == 0:
    print("motiontrakcer paralel init")
graspnet = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                        cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04]).cuda()
graspnet = torch.nn.parallel.DistributedDataParallel(graspnet, device_ids=[cfgs.local_rank], broadcast_buffers=False,
                                                      output_device=cfgs.local_rank, find_unused_parameters=False)

if cfgs.local_rank == 0:
    print("motiontrakcer init")
# Load the Adam optimizer
optimizer = optim.Adam(tracker.parameters(), lr=cfgs.learning_rate)
CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0)

# Load model checkpoint
start_epoch = 0

checkpoint_1 = torch.load(CHECKPOINT_PATH_1, map_location='cuda:{}'.format(cfgs.local_rank))
new_state_dict = OrderedDict()
for k, v in checkpoint_1['model_state_dict'].items():
    name = 'module.' + k if not k.startswith('module.') else k
    new_state_dict[name] = v
    
LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'w+')
LOG_FOUT.write(str(cfgs) + '\n')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

graspnet.load_state_dict(new_state_dict)
epoch = checkpoint_1['epoch']
if cfgs.local_rank == 0:
    log_string("-> loaded checkpoint of graspnet %s (epoch: %d)"%(CHECKPOINT_PATH_1, epoch))

if CHECKPOINT_PATH_2 is not None and os.path.isfile(CHECKPOINT_PATH_2):
    checkpoint_2 = torch.load(CHECKPOINT_PATH_2, map_location='cuda:{}'.format(cfgs.local_rank))
    tracker.load_state_dict(checkpoint_2['model_state_dict'])
    optimizer.load_state_dict(checkpoint_2['optimizer_state_dict'])
    start_epoch = checkpoint_2['epoch']
    if cfgs.local_rank == 0:
        log_string("-> loaded checkpoint of grasp tracker %s (epoch: %d)" % (CHECKPOINT_PATH_2, start_epoch))

def get_current_lr(epoch):
    lr = cfgs.learning_rate
    if epoch > 0 and epoch % 1 == 0:
        lr = lr * 0.5
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# TensorBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train_{}').format(cfgs.log_dir.split('/')[-1]))

def train_one_epoch():
    graspnet.eval()
    tracker.train()
    stat_dict = {}  # collect statistics

    batch_interval = 1
    for batch_idx, batch_data in enumerate(TRAIN_DATALOADER):
        for i in batch_data:
            for key in i.keys():
                i[key] = i[key].cuda()

        # forward pass
        for frame_id in range(0, cfgs.sequence_size):
            with torch.no_grad():
                end_points = graspnet(batch_data[frame_id])

            if 'batch_grasp_preds' in end_points.keys():
                grasp_preds = end_points['batch_grasp_preds']
            else:
                grasp_preds, end_points = pred_decode(end_points, remove_background=False) # (B*2, Ns, 17)
                end_points['batch_grasp_preds'] = grasp_preds

            if frame_id == 0:
                end_points_1 = copy.deepcopy(end_points)
                tracker(frame_id, grasp_preds, end_points_1)
                continue
            else:
                corr_pred_coarse, corr_pred_fine, training_mask, trans, rot, pose_label = tracker(frame_id, grasp_preds, end_points)
                loss, end_points = get_loss(merge_frame_end_points([end_points_1, end_points]), corr_pred_fine, corr_pred_coarse, training_mask, trans, rot, pose_label, frame_id, cfgs.rot_type, nfr=5)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=tracker.parameters(), max_norm=10)
            optimizer.step()

            for key in end_points:
                if ('loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key):
                    global_values = [torch.zeros_like(end_points[key]) for _ in range(cfgs.gpu_num)]
                    dist.all_gather(global_values, end_points[key])
                    if cfgs.local_rank == 0:
                        if key not in stat_dict:
                            stat_dict[key] = 0
                        stat_dict[key] += sum(global_values)
        tracker.module.clear()

        if (batch_idx + 1) % batch_interval == 0:
            if cfgs.local_rank == 0:
                log_string(' ----epoch: %03d  ---- batch: %03d ----' % (EPOCH_CNT, batch_idx + 1))
                log_string('experiment names: {}'.format(cfgs.log_dir.split('/')[-1]))
                for key in sorted(stat_dict.keys()):
                    TRAIN_WRITER.add_scalar(key, stat_dict[key]/(cfgs.gpu_num*(cfgs.sequence_size-1)*batch_interval), 
                                            (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * cfgs.batch_size)
                    log_string('mean %s: %f' % (key, stat_dict[key] / (cfgs.gpu_num*(cfgs.sequence_size-1)*batch_interval)))
                    stat_dict[key] = 0

def train(start_epoch):
    global EPOCH_CNT

    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        if cfgs.local_rank == 0:
            log_string('**** EPOCH %03d ****' % epoch)
            log_string('Current learning rate: %f' % (get_current_lr(epoch)))
            log_string(str(datetime.now()))

        np.random.seed()
        with torch.autograd.set_detect_anomaly(True):
            train_one_epoch()

        CosineLR.step()
        save_dict = {'epoch': epoch + 1, 'optimizer_state_dict': optimizer.state_dict(),
                     'model_state_dict': tracker.state_dict()}

        if dist.get_rank() == 0:
            torch.save(save_dict, os.path.join(cfgs.log_dir, 'tracker' + '_epoch_' + str(epoch + 1).zfill(2) + '.tar'))


if __name__ == '__main__':
    train(start_epoch)