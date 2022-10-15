import numpy as np
import skfuzzy as fuzz
import math
import dgl
import torch as th

def FuzzyInferSys(x1=None, x2=None, rel=None):
    # TODO, range should be modified
    # Relationship 0: robot and obstacles, altogether 3
    # x1, x2, big medium small
    # if x1 is big, x2 is big, then no need to worry, good, which means can igonre
    # if x1 is medium, x2 is medium, need to worry moderately, medium, which means long term planning
    # if x1 is small, x2 is small, then very dangerous, bad condition, which means short term planning to aviod the obstacle

    if rel == 0 or rel == 1:
        x1_range = np.arange(0, 11, 1)
        x2_range = np.arange(0, 11, 1)

        x11 = fuzz.trimf(x1_range, [0, 0, 5])
        x12 = fuzz.trimf(x1_range, [0, 5, 10])
        x13 = fuzz.trimf(x1_range, [5, 10, 10])

        x11_level = fuzz.interp_membership(x1_range, x11, x1)
        x12_level = fuzz.interp_membership(x1_range, x12, x1)
        x13_level = fuzz.interp_membership(x1_range, x13, x1)

        x21 = fuzz.trimf(x2_range, [0, 0, 5])
        x22 = fuzz.trimf(x2_range, [0, 5, 10])
        x23 = fuzz.trimf(x2_range, [5, 10, 10])

        x21_level = fuzz.interp_membership(x2_range, x21, x2)
        x22_level = fuzz.interp_membership(x2_range, x22, x2)
        x23_level = fuzz.interp_membership(x2_range, x23, x2)

        truth_value = th.stack((th.min(th.tensor(x11_level), th.tensor(x21_level)),
                                th.min(th.tensor(x12_level), th.tensor(x22_level)),
                                th.min(th.tensor(x13_level), th.tensor(x23_level))), dim=1)
                                # here, dim=0 --> 72 * 3 * 6
                                # here, dim=1 --> 72 * 6 * 3

        return truth_value.float()

    elif rel == 2 or rel == 3:
        x1_range = np.arange(0, 11, 1)

        x11 = fuzz.trimf(x1_range, [0, 0, 5])
        x12 = fuzz.trimf(x1_range, [0, 5, 10])
        x13 = fuzz.trimf(x1_range, [5, 10, 10])

        x11_level = fuzz.interp_membership(x1_range, x11, x1)
        x12_level = fuzz.interp_membership(x1_range, x12, x1)
        x13_level = fuzz.interp_membership(x1_range, x13, x1)

        truth_value = th.stack((th.tensor(x11_level), th.tensor(x12_level), th.tensor(x13_level)), dim=1)
        return truth_value.float()

def graph_and_fuzzy(node_infos): # generate graph, etypes, and truth values based on given node infos
    if node_infos.dim() == 2:
        node_infos = node_infos.unsqueeze(0)

    edge_src = []
    edge_dst = []
    edge_types = []
    truth_values = []
    node_num = 9

    for i in range(node_num):
        for j in range(node_num):

            if i == j:
                continue

            edge_src.append(i)
            edge_dst.append(j)
            '''
            relationships: 
            0: robot-target, 1: robot-obstacle
            2: target-obstacle, 3:obstacle-obstacle
            '''
            # robot-target
            if (i==0 and j==1) or (i==1 and j==0):
                edge_types.append(0)
                truth_values.append(nodes2tv(node_infos[:, 0], node_infos[:, 1], rel=0))

            # robot-obstacle
            elif (i==0 and 2<=j<=8):
                edge_types.append(1)
                truth_values.append(nodes2tv(node_infos[:, 0], node_infos[:, j], rel=1))

            # obstacle-robot
            elif (2<=i<=8 and j==0):
                edge_types.append(1)
                truth_values.append(nodes2tv(node_infos[:, 0], node_infos[:, i], rel=1))

            # target-obstacle
            elif (i==1 and 2<=j<=8) or (2<=i<=8 and j==1):
                edge_types.append(2)
                truth_values.append(nodes2tv(node_infos[:, i], node_infos[:, j], rel=2))

            else:
                edge_types.append(3)
                truth_values.append(nodes2tv(node_infos[:, i], node_infos[:, j], rel=3))

    return dgl.graph((edge_src, edge_dst)), th.tensor(edge_types), th.stack(truth_values, dim=0)

def nodes2tv(node1, node2, rel): # provide truth values based on two given nodes info
    # node1 always refer to moving object or satatic object while node2 refer to static object
    if rel == 0 or rel == 1:
        pos_robot = node1[:, :2]
        ori = node1[:, 2: 4]
        vel = node1[:, 4: 6]
        pot_target_or_obstacle = node2[:, :2]
        alpha = pot_target_or_obstacle - pos_robot
        beta = vel + ori
        # here need to think about batch process
        x1 = np.linalg.norm((pos_robot - pot_target_or_obstacle), axis=1)
        x2 = angle(alpha, beta)
        return FuzzyInferSys(x1=x1, x2=x2, rel=rel)

    elif rel == 2 or rel == 3:
        pos1 = node1[:, :2]
        pos2 = node2[:, :2]
        x1 = np.linalg.norm((pos1 - pos2), axis=1) # distance 
        return FuzzyInferSys(x1=x1, rel=rel)

def obs_to_feat(obs): # transfer observation into node features form
    # obs_size = 6 + 2 + 7*2 = 22
    
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)

    if obs.dim() == 2:
        m = th.nn.ZeroPad2d((0, 4, 0, 0))
        robot_info = obs[:, :6]
        target_info = m(obs[:, 6: 8])
        obstacle_infos = m(obs[:, 8:].view(-1, 7, 2))

        node_infos = th.cat((robot_info.unsqueeze(1), target_info.unsqueeze(1), obstacle_infos), dim=1)
        return node_infos

def angle(v1, v2): # calculate angle between two give vectors
    cos = th.nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_value = cos(v1, v2)
    return np.degrees(th.acos(cos_value))
