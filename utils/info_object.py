import numpy as np
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval

obj_names = ['cracker_box',
          'tomato_soup_can',
          'sugar_box',
          'mustard_bottle',
          'potted_meat_can',
          'banana',
          'bowl',
          'mug',
          'power_drill',
          'scissors',
          'chips_can',
          'strawberry',
          'apple',
          'lemon',
          'peach',
          'pear',
          'orange',
          'plum',
          'knife',
          'phillips_screwdriver',
          'flat_screwdriver',
          'racquetball',
          'b_cups',
          'd_cups',
          'a_toy_airplane',
          'c_toy_airplane',
          'd_toy_airplane',
          'f_toy_airplane',
          'h_toy_airplane',
          'i_toy_airplane',
          'j_toy_airplane',
          'k_toy_airplane',
          'padlock',
          'dragon',
          'secret_repair',
          'jvr_cleaning_foam',
          'dabao_wash_soup',
          'nzskincare_mouth_rinse',
          'dabao_sod',
          'soap_box',
          'kispa_cleanser',
          'darlie_toothpaste',
          'nivea_men_oil_control',
          'baoke_marker',
          'hosjam',
          'pitcher_cap',
          'dish',
          'white_mouse',
          'camel',
          'deer',
          'zebra',
          'large_elephant',
          'rhinocero',
          'small_elephant',
          'monkey',
          'giraffe',
          'gorilla',
          'weiquan',
          'darlie_box',
          'soap',
          'black_mouse',
          'dabao_facewash',
          'pantene',
          'head_shoulders_supreme',
          'thera_med',
          'dove',
          'head_shoulders_care',
          'lion',
          'coconut_juice_box',
          'hippo',
          'tape',
          'rubiks_cube',
          'peeler_cover',
          'peeler',
          'ice_cube_mould',
          'bar_clamp',
          'climbing_hold',
          'endstop_holder',
          'gearbox',
          'mount1',
          'mount2',
          'nozzle',
          'part1',
          'part3',
          'pawn',
          'pipe_connector',
          'turbine_housing',
          'vase']

def select_valid_grasp(gg, segs):
    obj_idxs = list(np.unique(segs))
    grasp_group = gg#.grasp_group_array
    gg_per_obj = []
    segs_obj = []
    gg_indices = []
    for id in obj_idxs:
        id_mask = (segs == id)
        gg_obj = grasp_group[id_mask]
        gg_obj_idxs = np.where((id_mask==True))[0]

        if np.sum(id_mask) < 10:
            gg_per_obj.append(gg_obj)
            segs_obj.append(segs[id_mask])
            gg_indices.append(gg_obj_idxs) # grasp index in ordinary group
        else:
            score = gg_obj.scores
            score_indices = np.array(sorted(range(len(score)), key=lambda k:score[k], reverse=True))
            gg_per_obj.append(gg_obj[score_indices[:10]])
            segs_obj.append(segs[score_indices[:10]])
            gg_indices.append(gg_obj_idxs[score_indices[:10]])

        
    gg_per_obj = GraspGroup(np.concatenate(gg_per_obj, axis=0))
    segs_obj = np.concatenate(segs_obj, axis=0)
    gg_indices = np.concatenate(gg_indices, axis=0)
    return gg_per_obj, segs_obj, gg_indices