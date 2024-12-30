import os
import numpy as np
from tqdm import tqdm
import open3d as o3d
import json
import open3d_plus as o3dp
import yaml


def _isArrayLike(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


SCENE_DIR_NAME = "scenes"
CAMERA_PREFIX = "camera_"
SCENE_METADATA_FILE = "metadata.json"
DATASET_METADATA_FILE = "default.yaml"
MODELS_DIR = "models"
GRASP_DIR = "grasp_label"
MULTIPLE_OBJECT_COLLISION_DIR_NAME = "collision_label_multiobj_25frame"
INITIAL_FRAME_COLLISION_DIR_NAME = "collision_label_initial_frame"
POSE_DIR = "pose"
COLOR_DIR = "color"
DEPTH_DIR = "depth"
REALSENSE_D_SERIALS_DEPTH_SCALE = 1000.0  # scale is 1000 for other cameras
REALSENSE_L_SERIALS_DEPTH_SCALE = 4000.0  # scale is 4000 for L515
Z_DISTANCE_THRESH = 2.0  # max distance from the point to the camera in z direction
GRASP_HEIGHT = 0.02


class MotionGrasp:
    def __init__(self, root, type=None):
        self.root = root
        self.scenes_root = os.path.join(self.root, SCENE_DIR_NAME)
        self.scene_name_list = self._get_scene_name_list()
        self.collision_path_dict_initial_frame = self._get_gt_collision_frame_dict(
            "initial_frame", type
        )
        self._load_camera_intrinsics()
        self._load_object_name_list()

    def _get_scene_name_list(self):
        return sorted(os.listdir(self.scenes_root))

    def _get_camera_sn_list(self, scene_name):
        scene_dir = os.path.join(self.scenes_root, scene_name)
        cam_sn_list = [
            name.split("_")[1]
            for name in os.listdir(scene_dir)
            if name.startswith(CAMERA_PREFIX)
        ]
        return sorted(cam_sn_list)

    def _check_metadata(self, scene_name, metadata):
        cam_sn_list = self._get_camera_sn_list(scene_name=scene_name)
        cam_set = set(cam_sn_list + ["num_" + cam_sn for cam_sn in cam_sn_list])
        key_set = set(metadata.keys())
        return cam_set == key_set

    def load_scene_metadata(self, scene_name):
        scene_dir = os.path.join(self.scenes_root, scene_name)
        with open(
            os.path.join(scene_dir, SCENE_METADATA_FILE), "r"
        ) as metadata_json_file:
            scene_meta = json.load(metadata_json_file)
        return scene_meta

    def _get_valid_frame_list(self, scene_name, camera_sn):
        scene_meta = self.load_scene_metadata(scene_name=scene_name)
        assert len(scene_meta[camera_sn]) == scene_meta["num_" + camera_sn]
        return [str(int_camera_sn) for int_camera_sn in scene_meta[camera_sn]]

    def get_frame_list(self, scene_name, camera_sn):
        camera_dir = os.path.join(
            self.root, SCENE_DIR_NAME, scene_name, CAMERA_PREFIX + camera_sn
        )
        return sorted(
            [
                file_name.replace(".png", "")
                for file_name in os.listdir(os.path.join(camera_dir, "color"))
            ]
        )

    def _load_camera_intrinsics(self):
        camera_sn_list = [
            camera_sn.replace(".npy", "")
            for camera_sn in os.listdir(os.path.join(self.root, "cam_intrinsics"))
        ]
        camera_intrinsics = dict()
        for camera_sn in camera_sn_list:
            camera_intrinsics[camera_sn] = np.load(
                os.path.join(self.root, "cam_intrinsics", "{}.npy".format(camera_sn))
            )
        self.camera_intrinsics = camera_intrinsics

    def load_cam_intrinsic(self, camera_sn):
        return self.camera_intrinsics[camera_sn]

    def get_rgb_path(self, scene_name, camera_sn, frame):
        return os.path.join(
            self.root,
            SCENE_DIR_NAME,
            scene_name,
            "{}{}".format(CAMERA_PREFIX, camera_sn),
            "color",
            "{}.png".format(frame),
        )

    def get_depth_path(self, scene_name, camera_sn, frame):
        return os.path.join(
            self.root,
            SCENE_DIR_NAME,
            scene_name,
            "{}{}".format(CAMERA_PREFIX, camera_sn),
            "depth",
            "{}.png".format(frame),
        )

    def load_point_cloud(self, scene_name, camera_sn, frame):
        depth_path = self.get_depth_path(scene_name, camera_sn, frame)
        rgb_path = self.get_rgb_path(scene_name, camera_sn, frame)
        intrinsic = self.load_cam_intrinsic(CAMERA_PREFIX+camera_sn)
        depth_scale = (
            REALSENSE_L_SERIALS_DEPTH_SCALE
            if camera_sn.startswith("f")
            else REALSENSE_D_SERIALS_DEPTH_SCALE
        )
        pcd = o3dp.generate_scene_pointcloud(
            depth=depth_path,
            rgb=rgb_path,
            intrinsics=intrinsic,
            depth_scale=depth_scale,
        )
        points, colors = o3dp.pcd2array(pcd)
        mask = points[:, 2] < Z_DISTANCE_THRESH
        return o3dp.array2pcd(points[mask], colors[mask])

    def _load_object_name_list(self):
        with open(
            os.path.join(self.root, DATASET_METADATA_FILE), "r"
        ) as scene_metadata_file:
            scene_metadata_yaml = scene_metadata_file.read()
        scene_metadata = yaml.load(scene_metadata_yaml, Loader=yaml.FullLoader)
        self.object_name_list = [
            object_file_name.replace(".ply", "")
            for object_file_name in scene_metadata["object_model_list"]
        ]

    def load_object_point_cloud(self, obj_id):
        models_dir = os.path.join(self.root, MODELS_DIR)
        return o3d.io.read_point_cloud(
            os.path.join(models_dir, "{}.ply".format(self.object_name_list[obj_id]))
        )

    def load_scene_object_list(self, scene_name, camera_sn, frame):
        frame_pose_dir = os.path.join(
            self.root,
            SCENE_DIR_NAME,
            scene_name,
            "{}{}".format(CAMERA_PREFIX, camera_sn),
            POSE_DIR,
            frame,
        )
        return list(
            set(
                [
                    int(obj_id.replace(".npy", "").replace("registered_", ""))
                    for obj_id in os.listdir(frame_pose_dir)
                    if obj_id.endswith(".npy")
                ]
            )
        )

    def load_scene_registered_object_list(self, scene_name, camera_sn, frame):
        frame_pose_dir = os.path.join(
            self.root,
            SCENE_DIR_NAME,
            scene_name,
            "{}{}".format(CAMERA_PREFIX, camera_sn),
            POSE_DIR,
            frame,
        )

        return list(
            set(
                [
                    int(obj_id.replace(".npy", "").replace("registered_", ""))
                    for obj_id in os.listdir(frame_pose_dir)
                    if (obj_id.startswith("registered_") and obj_id.endswith(".npy"))
                ]
            )
        )

    def load_object_pose(self, scene_name, camera_sn, frame, obj_id, registered=True):
        frame_pose_dir = os.path.join(
            self.root,
            SCENE_DIR_NAME,
            scene_name,
            "{}{}".format(CAMERA_PREFIX, camera_sn),
            POSE_DIR,
            frame,
        )
        prefix = "" if not registered else "registered_"
        if not os.path.exists(
            os.path.join(frame_pose_dir, "{}{}.npy".format(prefix, obj_id))
        ):
            return None
        pose = np.load(os.path.join(frame_pose_dir, "{}{}.npy".format(prefix, obj_id)))
        return pose

    def get_registered_object_pose_filename(self, scene_name, camera_sn, frame, obj_id):
        frame_pose_dir = os.path.join(
            self.root,
            SCENE_DIR_NAME,
            scene_name,
            "{}{}".format(CAMERA_PREFIX, camera_sn),
            POSE_DIR,
            frame,
        )
        return os.path.join(frame_pose_dir, "registered_{}.npy".format(obj_id))

    def load_scene_with_object(self, scene_name, camera_sn, frame, registered=True):
        pcds = [
            self.load_point_cloud(
                scene_name=scene_name, camera_sn=camera_sn, frame=frame
            )
        ]
        frame_pose_dir = os.path.join(
            self.root,
            SCENE_DIR_NAME,
            scene_name,
            "{}{}".format(CAMERA_PREFIX, camera_sn),
            POSE_DIR,
            frame,
        )
        with_pose_obj_ids = self.load_scene_object_list(scene_name, camera_sn, frame)
        for obj_id in with_pose_obj_ids:
            pcd = self.load_object_point_cloud(obj_id)
            pose = self.load_object_pose(
                scene_name, camera_sn, frame, obj_id, registered=registered
            )
            pcd.transform(pose)
            pcds.append(pcd)
        return pcds

    def get_near_frames(self, scene_name, camera_sn, frame, max_distance):
        near_frames = []
        frame_list = self.get_frame_list(scene_name=scene_name, camera_sn=camera_sn)
        if not frame in frame_list:
            raise ValueError("frame {} is not a valid frame in scene {} camera {}")
        for all_frame in frame_list:
            if (
                abs(int(frame) - int(all_frame)) < max_distance
                and not frame == all_frame
            ):
                near_frames.append(all_frame)
        return near_frames


    def _get_gt_collision_frame_dict(self, split="multiobj_25frame", type=None):
        assert split in [
            "multiobj_25frame",
            "initial_frame",
        ], "argument 'split' only support 'multiobj_25frame' and 'initial_frame'"
        collision_dir = (
            MULTIPLE_OBJECT_COLLISION_DIR_NAME
            if split == "multiobj_25frame"
            else INITIAL_FRAME_COLLISION_DIR_NAME
        )
        frame_path = type + '_frame_dicts.json' if type is not None else 'frame_dicts.json'
        with open(os.path.join(self.root, collision_dir, frame_path), "r") as f:
            frame_dict = json.load(f)

        # convert data type of obj_id (str -> int)
        if split == "initial_frame":
            for scene_name in frame_dict:
                for camera_sn in frame_dict[scene_name]:
                    frame_dict_camera = frame_dict[scene_name][camera_sn]
                    new_frame_dict_camera = {}
                    for obj_id in frame_dict_camera:
                        new_frame_dict_camera[int(obj_id)] = frame_dict_camera[obj_id]
                    frame_dict[scene_name][camera_sn] = new_frame_dict_camera

        return frame_dict


    def _load_initial_frame_collision_labels(
        self, scene_name, camera_sn, frame, obj_id
    ):
        frame_dict = self.collision_path_dict_initial_frame
        log_info = "%s contains no valid collision label, check self.collision_path_dict_initial_frame"
        assert scene_name in frame_dict, log_info % (scene_name)
        assert camera_sn in frame_dict[scene_name], log_info % (
            "%s/%s" % (scene_name, camera_sn)
        )
        assert obj_id in frame_dict[scene_name][camera_sn], log_info % (
            "object %d in %s/%s" % (obj_id, scene_name, camera_sn)
        )
        assert frame == frame_dict[scene_name][camera_sn][obj_id], (
            "incorrect initial frame of object %d, check self.collision_path_dict_initial_frame"
            % obj_id
        )

        collision_path = os.path.join(
            self.root,
            INITIAL_FRAME_COLLISION_DIR_NAME,
            scene_name,
            camera_sn,
            frame,
            "%d.npy" % obj_id,
        )
        collision = np.load(collision_path)

        return collision

    def get_camera_obj_ids(self, scene_name, camera_sn):
        obj_ids = set()
        frame_list = self.get_frame_list(scene_name=scene_name, camera_sn=camera_sn)
        for frame in frame_list:
            obj_ids = obj_ids.union(
                set(
                    self.load_scene_registered_object_list(
                        scene_name=scene_name, camera_sn=camera_sn, frame=frame
                    )
                )
            )
        return sorted(list(obj_ids))