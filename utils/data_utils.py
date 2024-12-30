""" Tools for data processing.
    Author: chenxi-wang
"""

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import numpy as np
from xmlhandler import xmlReader
from transforms3d.euler import euler2mat


class CameraInfo():
    """ Camera intrisics for point cloud creation. """

    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert (depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                points in original coordinates
            transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [np.ndarray, (N,3), np.float32]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = np.dot(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = np.ones(cloud.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([cloud, ones], axis=1)
        cloud_transformed = np.dot(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed

def compute_point_dists(A, B):
    """ Compute pair-wise point distances in two matrices.

        Input:
            A: [np.ndarray, (N,3), np.float32]
                point cloud A
            B: [np.ndarray, (M,3), np.float32]
                point cloud B

        Output:
            dists: [np.ndarray, (N,M), np.float32]
                distance matrix
    """
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    dists = np.linalg.norm(A - B, axis=-1)
    return dists

def remove_invisible_grasp_points(cloud, grasp_points, pose, th=0.01):
    """ Remove invisible part of object model according to scene point cloud.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                scene point cloud
            grasp_points: [np.ndarray, (M,3), np.float32]
                grasp point label in object coordinates
            pose: [np.ndarray, (4,4), np.float32]
                transformation matrix from object coordinates to world coordinates
            th: [float]
                if the minimum distance between a grasp point and the scene points is greater than outlier, the point will be removed

        Output:
            visible_mask: [np.ndarray, (M,), np.bool]
                mask to show the visible part of grasp points
    """
    grasp_points_trans = transform_point_cloud(grasp_points, pose)
    dists = compute_point_dists(grasp_points_trans, cloud)
    min_dists = dists.min(axis=1)
    visible_mask = (min_dists < th)
    return visible_mask

def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
    """ Keep points in workspace as input.

        Input:
            cloud: [np.ndarray, (H,W,3), np.float32]
                scene point cloud
            seg: [np.ndarray, (H,W,), np.uint8]
                segmantation label of scene points
            trans: [np.ndarray, (4,4), np.float32]
                transformation matrix for scene points, default: None.
            organized: [bool]
                whether to keep the cloud in image shape (H,W,3)
            outlier: [float]
                if the distance between a point and workspace is greater than outlier, the point will be removed
                
        Output:
            workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
                mask to indicate whether scene points are in workspace
    """
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape([h * w, 3])
        seg = seg.reshape(h * w)
    if trans is not None:
        cloud = transform_point_cloud(cloud, trans)
    foreground = cloud[seg > 0]
    
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)
    mask_x = ((cloud[:, 0] > xmin - outlier) & (cloud[:, 0] < xmax + outlier))
    mask_y = ((cloud[:, 1] > ymin - outlier) & (cloud[:, 1] < ymax + outlier))
    mask_z = ((cloud[:, 2] > zmin - outlier) & (cloud[:, 2] < zmax + outlier))
    workspace_mask = (mask_x & mask_y & mask_z)
    if organized:
        workspace_mask = workspace_mask.reshape([h, w])

    return workspace_mask

def parse_posevector(posevector):
    '''
    **Input:**
    - posevector: list of pose
    **Output:**
    - obj_idx: int of the index of object.
    - mat: numpy array of shape (4, 4) of the 6D pose of object.
    '''
    mat = np.zeros([4,4],dtype=np.float32)
    alpha, beta, gamma = posevector[4:7]
    alpha = alpha / 180.0 * np.pi
    beta = beta / 180.0 * np.pi
    gamma = gamma / 180.0 * np.pi
    mat[:3,:3] = euler2mat(alpha, beta, gamma)
    mat[:3,3] = posevector[1:4]
    mat[3,3] = 1
    obj_idx = int(posevector[0])
    return obj_idx, mat

def read_pose_from_xml(pose_xml_path, extri_matrix=np.eye(4)):
    scene_reader = xmlReader(pose_xml_path)
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    pose_list = []
    for posevec in posevectors:
        obj_idx, mat = parse_posevector(posevec)  # pose in world?
        # if pose_mat is in world, we have to transform it in camera with extrinsics
        # extrinsics: RT_cam_in_W;   pose_mat: RT_obj_in_W,  pose_rtn: RT_obj_in_cam
        # mat = matmul(inv(extrinsics), pose_mat)

        obj_list.append(obj_idx)
        pose_list.append(np.matmul(np.linalg.inv(extri_matrix), mat))
    return np.array(pose_list)  # [(4,4),...]


# def read_pose_from_xml(pose_xml_path, extri_matrix=np.eye(4)):
#     scene_reader = xmlReader(pose_xml_path)
#     posevectors = scene_reader.getposevectorlist()
#     obj_list = []
#     pose_list = []
#     for posevec in posevectors:
#         obj_idx, mat = parse_posevector(posevec)  # pose in world?
#         # if pose_mat is in world, we have to transform it in camera with extrinsics
#         # extrinsics: RT_cam_in_W;   pose_mat: RT_obj_in_W,  pose_rtn: RT_obj_in_cam
#         # mat = matmul(inv(extrinsics), pose_mat)

#         obj_list.append(obj_idx)
#         pose_list.append(np.matmul(np.linalg.inv(extri_matrix), mat))
#     return obj_list, pose_list  # [(4,4),...]

def read_obj_id_from_xml(pose_xml_path):
    scene_reader = xmlReader(pose_xml_path)
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    for posevec in posevectors:
        obj_idx, _= parse_posevector(posevec)  # pose in world?
        # if pose_mat is in world, we have to transform it in camera with extrinsics
        # extrinsics: RT_cam_in_W;   pose_mat: RT_obj_in_W,  pose_rtn: RT_obj_in_cam
        # mat = matmul(inv(extrinsics), pose_mat)

        obj_list.append(obj_idx)
    return np.array(obj_list)  # [(4,4),...]

def find_obj_name(obj_dict, model_id):
    object_name = [k for k, v in obj_dict.items() if v == model_id]
    return object_name[0]


obj_dict = {'cracker_box':0,
            'tomato_soup_can':1,
            'sugar_box':2,
            'mustard_bottle':3,
            'potted_meat_can':4,
            'banana':5,
            'bowl':6,
            'mug':7,
            'power_drill':8,
            'scissors':9,
            'chips_can':10,
            'strawberry':11,
            'apple':12,
            'lemon':13,
            'peach':14,
            'pear':15,
            'orange':16,
            'plum':17,
            'knife':18,
            'phillips_screwdriver':19,   # start
            'flat_screwdriver':20,
            'racquetball':21,
            'b_cups':22,
            'd_cups':23,
            'a_toy_airplane':24,
            'c_toy_airplane':25,
            'd_toy_airplane':26,
            'f_toy_airplane':27,
            'h_toy_airplane':28,
            'i_toy_airplane':29,
            'j_toy_airplane':30,
            'k_toy_airplane':31,
            'padlock':32,
            'dragon':33,
            'secret_repair':34,
            'jvr_cleaning_foam':35,
            'dabao_wash_soup':36,
            'nzskincare_mouth_rinse':37,
            'dabao_sod':38,
            'soap_box':39,               #end
            'kispa_cleanser':40,
            'darlie_toothpaste':41,
            'nivea_men_oil_control':42,
            'baoke_marker':43,
            'hosjam':44,
            'pitcher_cap':45,     ## lost
            'dish':46,
            'white_mouse':47,
            'camel':48,
            'deer':49,
            'zebra':50,           ##start
            'large_elephant':51,  
            'rhinocero':52,       
            'small_elephant':53,  
            'monkey':54,          
            'giraffe':55,         
            'gorilla':56,         
            'weiquan':57,         
            'darlie_box':58,      
            'soap':59,            
            'black_mouse':60,     
            'dabao_facewash':61,
            'pantene':62,         
            'head_shoulders_supreme':63,   
            'thera_med':64,       
            'dove':65,
            'head_shoulders_care':66,      
            'lion':67,            
            'coconut_juice_box':68,        
            'hippo':69,           
            'tape':70,            ##end
            'rubiks_cube':71,
            'peeler_cover':72,
            'peeler':73,
            'ice_cube_mould':74,
            'bar_clamp':75,
            'climbing_hold':76,
            'endstop_holder':77,
            'gearbox':78,
            'mount1':79,
            'mount2':80,
            'nozzle':81,
            'part1':82,
            'part3':83,
            'pawn':84,
            'pipe_connector':85,
            'turbine_housing':86,
            'vase':87}