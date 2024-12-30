import os
import math
import sys
import numpy as np
import trackeval
import json
from itertools import product
import copy

from .motiongrasp import MotionGrasp, SCENE_DIR_NAME, CAMERA_PREFIX
from graspnetAPI.grasp import Grasp, GraspGroup

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from utils.eval_utils import npy_2_grasp


QUERY_GRASP_POSE_DIR_NAME = "query_grasp_poses"
QUERY_GRASP_POSE_FILE_NAME = "grasp_group.npy"
QUERY_GRASP_POSE_JSON_NAME = "grasp_query.json"
START_FRAME_KEY_NAME = "start_frame"
OBJECT_ID_KEY_NAME = "object_id"


class GraspDist:
    @classmethod
    def dist_from_grasp_pose(cls, g1: Grasp, g2: Grasp):

        translation = np.linalg.norm(g1.translation - g2.translation)
        trace = np.matmul(g1.rotation_matrix, g2.rotation_matrix.T).trace()
        trace = min(max(trace, 1.0), 3.0)
        rotation = math.acos(0.5 * (trace - 1.0))
        return cls(translation, rotation)

    def __init__(self, translation=0.1, rotation=30.0 / 180.0 * np.pi):
        self.translation = translation
        self.rotation = rotation

    def __lt__(self, other):
        """less than a thresh."""
        return self.translation <= other.translation and self.rotation <= other.rotation

    def __gt__(self, other):
        """greater than a thresh."""
        return self.translation > other.translation or self.rotation > other.rotation

    def __repr__(self):
        return "GraspDist:(t: {}, r:{})".format(self.translation, self.rotation)


class MotionGraspEval(MotionGrasp):
    def __init__(self, root, pred_dir, log_dir=None, type=None, trans=0.1, rot=30.0/180.0*np.pi):
        super(MotionGraspEval, self).__init__(root, type)
        self.pred_dir = pred_dir
        global QUERY_GRASP_POSE_DIR_NAME
        if log_dir is not None:
            QUERY_GRASP_POSE_DIR_NAME = QUERY_GRASP_POSE_DIR_NAME + log_dir
        self.dist_thresh = GraspDist(trans, rot)

    def _eval_mgta(self, data):
        metric = trackeval.metrics.CLEAR()
        metric_result = metric.eval_sequence(data)
        return metric_result

    def _parse_tracker(self, scene_name, camera_sn):
        frames = self.get_frame_list(scene_name=scene_name, camera_sn=camera_sn)
        num_time_step = len(frames)
        max_pred = 0
        for frame in frames:
            tracker_ids = [
                int(pred_file.split(".")[0])
                for pred_file in os.listdir(
                    os.path.join(self.pred_dir, scene_name, CAMERA_PREFIX+camera_sn, frame)
                )
            ]
            if len(tracker_ids) > 0:
                max_pred = max(max_pred, max(tracker_ids)+1)
        tracker_present = np.zeros((num_time_step, max_pred), dtype=bool)
        tracker_pose_dict = dict()

        for time_step, frame in enumerate(frames):
            tracker_pose_dict[time_step] = dict()
            for tracker_id in range(max_pred):
                pose_file = os.path.join(
                    self.pred_dir,
                    scene_name,
                    CAMERA_PREFIX+camera_sn,
                    frame,
                    "{}.npy".format(tracker_id),
                )
                if os.path.exists(pose_file):
                    tracker_present[time_step][tracker_id] = True
                    tracker_pose_dict[time_step][tracker_id] = npy_2_grasp(
                        pose_file
                    )
        return max_pred, tracker_present, tracker_pose_dict

    def _generate_gt(self, scene_name, camera_sn, dump_dir=None):
        frame_list = self.get_frame_list(scene_name=scene_name, camera_sn=camera_sn)
        num_time_step = len(frame_list)
        query_grasp_pose_annotation_dir = os.path.join(
            self.root,
            SCENE_DIR_NAME,
            scene_name,
            "{}{}".format(CAMERA_PREFIX, camera_sn),
            QUERY_GRASP_POSE_DIR_NAME,
        )
        query_grasp_poses = GraspGroup().from_npy(
            os.path.join(
                query_grasp_pose_annotation_dir,
                QUERY_GRASP_POSE_FILE_NAME,
            )
        )
        json_path = os.path.join(
            query_grasp_pose_annotation_dir, QUERY_GRASP_POSE_JSON_NAME
        )
        with open(json_path, "r") as grasp_info_f:
            grasp_info_dict = json.load(grasp_info_f)
        num_gt_ids = len(query_grasp_poses)
        gt_present = np.zeros((num_time_step, num_gt_ids), dtype=bool)
        gt_pose_dict = dict()

        for gt_id, start_grasp_pose in enumerate(query_grasp_poses):
            start_frame = grasp_info_dict[gt_id][START_FRAME_KEY_NAME]
            attached_object_id = grasp_info_dict[gt_id][OBJECT_ID_KEY_NAME]
            find_start_frame_flag = False
            start_frame = str(start_frame).zfill(4)
            for time_step in range(num_time_step):
                if not frame_list[time_step] == start_frame:
                    continue
                start_time_step = time_step
                find_start_frame_flag = True
                break
            if not find_start_frame_flag:
                raise ValueError("Start frame not found")
            start_pose = self.load_object_pose(
                scene_name, camera_sn, start_frame, attached_object_id, registered=True
            )

            if start_pose is None:
                continue

            one = np.zeros([1,4])
            one[0,3] = 1
            start_pose = np.concatenate([start_pose, one], axis=0)
            for time_step in range(start_time_step, num_time_step):
                if not time_step in gt_pose_dict.keys():
                    gt_pose_dict[time_step] = dict()
                frame = frame_list[time_step]
                current_pose = self.load_object_pose(
                    scene_name, camera_sn, frame, attached_object_id, registered=True
                )
                if current_pose is None:
                    continue

                current_pose = np.concatenate([current_pose, one], axis=0)

  
                start_grasp_pose_ = copy.deepcopy(start_grasp_pose)
                current_grasp_pose = start_grasp_pose_.transform(
                    np.linalg.inv(start_pose)
                ).transform(
                    current_pose
                )

                gt_present[time_step][gt_id] = True
                gt_pose_dict[time_step][gt_id] = copy.deepcopy(current_grasp_pose)

        if dump_dir is not None:
            scene_dir = os.path.join(dump_dir, scene_name)
            camera_dir = os.path.join(scene_dir, CAMERA_PREFIX+camera_sn)
            os.makedirs(camera_dir, exist_ok=True)
            for time_step in range(num_time_step):
                frame = frame_list[time_step]
                frame_dir = os.path.join(camera_dir, frame)
                os.makedirs(frame_dir, exist_ok=True)
                for gt_id in range(num_gt_ids):
                    if gt_present[time_step][gt_id]:
                        np.save(os.path.join(frame_dir, "{}.npy".format(gt_id)), 
                                gt_pose_dict[time_step][gt_id].grasp_array)
        return num_gt_ids, gt_present, gt_pose_dict

    def _calculate_similarity(
        self, tracker_present, tracker_pose_dict, gt_present, gt_pose_dict
    ):

        num_time_step_tracker, num_tracker_ids = tracker_present.shape
        num_time_step_gt, num_gt_ids = gt_present.shape
        assert (
            num_time_step_tracker == num_time_step_gt
        ), "time steps in tracker and gt should be the same"
        num_time_step = num_time_step_gt
        similarity = np.zeros(
            shape=(num_time_step, num_gt_ids, num_tracker_ids), dtype=np.uint8
        )
        for time_step, gt_id, tracker_id in product(
            range(num_time_step), range(num_gt_ids), range(num_tracker_ids)
        ):
            if tracker_present[time_step, tracker_id] and gt_present[time_step, gt_id]:
                gt_pose = gt_pose_dict[time_step][gt_id]
                tracker_pose = tracker_pose_dict[time_step][tracker_id]

                if (
                    GraspDist.dist_from_grasp_pose(gt_pose, tracker_pose)
                    < self.dist_thresh
                ):
                    similarity[time_step, gt_id, tracker_id] = 1

        return similarity

    def _parse_files(self, scene_name, camera_sn, gt_dir=None):
        frame_list = self.get_frame_list(scene_name=scene_name, camera_sn=camera_sn)
        num_timestep = len(frame_list)

        num_gt_ids, gt_present, gt_pose_dict = self._generate_gt(scene_name, camera_sn, dump_dir=gt_dir)
        num_tracker_ids, tracker_present, tracker_pose_dict = self._parse_tracker(
            scene_name, camera_sn
        )
        similarity = self._calculate_similarity(
            tracker_present, tracker_pose_dict, gt_present, gt_pose_dict
        )
        return (
            num_timestep,
            num_gt_ids,
            num_tracker_ids,
            gt_present,
            tracker_present,
            similarity,
        )

    def _from_dense(
        self,
        num_timesteps,
        num_gt_ids,
        num_tracker_ids,
        gt_present,
        tracker_present,
        similarity,
    ):

        gt_subset = [np.flatnonzero(gt_present[t, :]) for t in range(num_timesteps)]
        tracker_subset = [
            np.flatnonzero(tracker_present[t, :]) for t in range(num_timesteps)
        ]
        similarity_subset = [
            similarity[t][gt_subset[t], :][:, tracker_subset[t]]
            for t in range(num_timesteps)
        ]

        data = {
            "num_timesteps": num_timesteps,
            "num_gt_ids": num_gt_ids,
            "num_tracker_ids": num_tracker_ids,
            "num_gt_dets": np.sum(gt_present),
            "num_tracker_dets": np.sum(tracker_present),
            "gt_ids": gt_subset,
            "tracker_ids": tracker_subset,
            "similarity_scores": similarity_subset,
        }
        return data

    def get_seq_mgta(self, scene_name, camera_sn, gt_dir=None):
        data = self._from_dense(*(self._parse_files(scene_name, camera_sn, gt_dir)))
        return self._eval_mgta(data)

    def eval_mgta_all(self, gt_dir=None):
        mgta_list = []
        fn_list = []
        fp_list = []
        id_list = []
        mgta_dict = dict()
        frame_dict = self.collision_path_dict_initial_frame
        for scene_name in frame_dict.keys():
            # if int(scene_name.split('_')[-1]) >= 187: continue
            
            mgta_dict[scene_name] = dict()
            for camera_sn in frame_dict[scene_name].keys():
                mgta = self.get_seq_mgta(scene_name, camera_sn, gt_dir=gt_dir)
                mgta_list.append(mgta['MOTA'])
                fn_list.append(mgta['CLR_FN'])
                fp_list.append(mgta['CLR_FP'])
                id_list.append(mgta['IDSW'])
                mgta_dict[scene_name][camera_sn] = mgta['MOTA']
                print('\rProcessing: {}, camera_{}, mgta={}, FN={}, FP={}, IDSW={}' \
                        .format(scene_name, camera_sn, mgta['MOTA'], mgta['CLR_FN'], mgta['CLR_FP'], mgta['IDSW']), end='')

        avg_mgta = sum(mgta_list) / len(mgta_list)
        avg_fn = sum(fn_list) / len(fn_list)
        avg_fp = sum(fp_list) / len(fp_list)
        avg_id = sum(id_list) / len(id_list)
        print("===============")
        print('\naverage mgta: {}, fn:{}, tp:{}, idsw:{}'.format(avg_mgta, avg_fn, avg_fp, avg_id))

        return avg_mgta

    def load_query_pose(self, scene_name, camera_sn):
        if not scene_name in self.collision_path_dict_initial_frame.keys():
            raise ValueError("No valid query in scene:{}".format(scene_name))
        if not camera_sn in self.collision_path_dict_initial_frame[scene_name].keys():
            raise ValueError(
                "No valid query in scene:{}, camere:{}".format(scene_name, camera_sn)
            )
        query_dir = os.path.join(
            self.root,
            "scenes",
            scene_name,
            "{}{}".format(CAMERA_PREFIX, camera_sn),
            QUERY_GRASP_POSE_DIR_NAME,
        )
        query_json_path = os.path.join(query_dir, QUERY_GRASP_POSE_JSON_NAME)
        query_pose_path = os.path.join(query_dir, QUERY_GRASP_POSE_FILE_NAME)
        with open(query_json_path) as query_f:
            query_info = json.load(query_f)
        query_pose = GraspGroup(query_pose_path)
        num_grasp = len(query_info)
        if not len(query_pose) == num_grasp:
            raise ValueError("grasp number in json and GraspGroup doesn't match")
        query_list = []
        for i in range(num_grasp):
            query_list.append(
                {
                    "start_frame": query_info[i]["start_frame"],
                    "object_id": query_info[i]["object_id"],
                    "grasp_pose": query_pose[i],
                }
            )
        return query_list