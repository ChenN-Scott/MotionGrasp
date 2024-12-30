import numpy as np
from graspnetAPI.grasp import Grasp
from graspnetAPI.utils.eval_utils import collision_detection, transform_points, compute_point_distance, compute_closest_points, get_grasp_score, create_table_points, voxel_sample_points
from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory
from graspnetAPI.utils.config import get_config


def npy_2_grasp(npy_file_path):
    grasp_array = np.load(npy_file_path, allow_pickle=True)
    return Grasp(grasp_array)


def eval_grasp_all(grasp_group, models, dexnet_models, poses, config, table=None, voxel_size=0.008, TOP_K = 50):
    num_models = len(models)
    ## grasp nms
    grasp_group = grasp_group.nms(0.03, 30.0/180*np.pi)

    ## assign grasps to object
    # merge and sample scene
    model_trans_list = list()
    seg_mask = list()
    for i,model in enumerate(models):
        model_trans = transform_points(model, poses[i])
        seg = i * np.ones(model_trans.shape[0], dtype=np.int32)
        model_trans_list.append(model_trans)
        seg_mask.append(seg)
    seg_mask = np.concatenate(seg_mask, axis=0)
    scene = np.concatenate(model_trans_list, axis=0)

    # assign grasps
    indices = compute_closest_points(grasp_group.translations, scene)
    model_to_grasp = seg_mask[indices]
    pre_grasp_list = list()
    pre_grasp_indices_list = list()
    for i in range(num_models):
        grasp_i = grasp_group[model_to_grasp==i]
        pre_grasp_list.append(grasp_i.grasp_group_array)

    all_grasp_list = np.vstack(pre_grasp_list)

    remain_mask = np.argsort(all_grasp_list[:,0])[::-1]
    min_score = all_grasp_list[remain_mask[min(49,len(remain_mask) - 1)],0]
    
    grasp_list = []
    for i in range(num_models):
        remain_mask_i = pre_grasp_list[i][:,0] >= min_score
        grasp_list.append(pre_grasp_list[i][remain_mask_i])
    # grasp_list = pre_grasp_list

    ## collision detection
    if table is not None:
        scene = np.concatenate([scene, table])

    collision_mask_list, empty_list, dexgrasp_list = collision_detection(
        grasp_list, model_trans_list, dexnet_models, poses, scene, outlier=0.05, return_dexgrasps=True)
    
    ## evaluate grasps
    # score configurations
    force_closure_quality_config = dict()
    fc_list = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    for value_fc in fc_list:
        value_fc = round(value_fc, 2)
        config['metrics']['force_closure']['friction_coef'] = value_fc
        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(config['metrics']['force_closure'])
    # get grasp scores
    score_list = list()
    
    for i in range(num_models):
        dexnet_model = dexnet_models[i]
        collision_mask = collision_mask_list[i]
        dexgrasps = dexgrasp_list[i]
        scores = list()
        num_grasps = len(dexgrasps)
        for grasp_id in range(num_grasps):
            if collision_mask[grasp_id]:
                scores.append(-1.)
                continue
            if dexgrasps[grasp_id] is None:
                scores.append(-1.)
                continue
            grasp = dexgrasps[grasp_id]
            score = get_grasp_score(grasp, dexnet_model, fc_list, force_closure_quality_config)
            scores.append(score)
        score_list.append(np.array(scores))

    return grasp_list, score_list, collision_mask_list

    
def eval_first_scene(ge, scene_id, frame_id, grasp_group, TOP_K = 50, return_list = False,vis = False, max_width = 0.1):
    config = get_config()
    table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)

    model_list, dexmodel_list, _ = ge.get_scene_models(scene_id, ann_id=0)

    model_sampled_list = list()
    for model in model_list:
        model_sampled = voxel_sample_points(model, 0.008)
        model_sampled_list.append(model_sampled)

    _, pose_list, camera_pose, align_mat = ge.get_model_poses(scene_id, frame_id)
    table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

    # clip width to [0,max_width]                
    gg_array = grasp_group.grasp_group_array
    min_width_mask = (gg_array[:,1] < 0)
    max_width_mask = (gg_array[:,1] > max_width)
    gg_array[min_width_mask,1] = 0
    gg_array[max_width_mask,1] = max_width
    grasp_group.grasp_group_array = gg_array

    grasp_list, score_list, collision_mask_list = eval_grasp_all(grasp_group, model_sampled_list, dexmodel_list, \
                                                                pose_list, config, table=table_trans, voxel_size=0.008, TOP_K = TOP_K)

    grasp_list = [x for x in grasp_list if len(x) != 0]
    score_list = [x for x in score_list if len(x) != 0]
    collision_mask_list = [x for x in collision_mask_list if len(x)!=0]
    
    grasp_list = np.array(grasp_list, dtype=object)[None, :]
    score_list = np.array(score_list, dtype=object)[None, :]
    collision_mask_list = np.array(collision_mask_list, dtype=object)[None, :]
    
    # sort in scene level
    grasp_confidence = grasp_list[:,0]
    indices = np.argsort(-grasp_confidence)
    grasp_list, score_list, collision_mask_list = grasp_list[indices], score_list[indices], collision_mask_list[indices]
    
    return grasp_list, score_list, collision_mask_list