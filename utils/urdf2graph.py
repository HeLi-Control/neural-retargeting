import torch
from torch_geometric.data import Data as OldData
from urdfpy import URDF, matrix_to_xyz_rpy
from configs.yumi_urdf_config import hand_cfg, yumi_cfg
import math


class Data(OldData):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "hand_edge_index":
            return self.hand_num_nodes
        else:
            return 0


def load_joints_from_urdf(urdf_file: str) -> tuple:
    # load URDF
    robot = URDF.load(urdf_file)
    # parse joint params
    joints = {}
    for joint in robot.joints:
        # joint attributes
        joints[joint.name] = {
            "type": joint.joint_type,
            "axis": joint.axis,
            "parent": joint.parent,
            "child": joint.child,
            "origin": matrix_to_xyz_rpy(joint.origin),
            "lower": joint.limit.lower if joint.limit else 0,
            "upper": joint.limit.upper if joint.limit else 0,
        }
    return robot, joints


def collect_edge_data_from_joints(joints: dict, joints_index: dict, cfg: dict) -> tuple:
    # collect edge index & edge feature
    edge_index = []
    edge_attr = []
    for edge in cfg["edges"]:
        parent, child = edge
        # add edge index
        edge_index.append(torch.LongTensor([joints_index[parent], joints_index[child]]))
        # add edge attr
        edge_attr.append(torch.Tensor(joints[child]["origin"]))
        # [tensor([0, 1]), tensor([1, 2]), tensor([2, 3]), tensor([3, 4]), ...]
    edge_index = torch.stack(edge_index, dim=0)  # tensor([[0, 1], [1, 2], ...])
    edge_index = edge_index.permute(1, 0)  # tensor([[0, 1, ...], [1, 2, ...]])
    edge_attr = torch.stack(edge_attr, dim=0)  # tensor([[xyz rpy], [xyz rpy], ...])
    return edge_index, edge_attr


def calc_distance(x, y, z) -> float:
    return math.sqrt(x**2 + y**2 + z**2)


def calc_root_distance(parent, joints_name, offset) -> torch.Tensor:
    # distance to root joint
    root_distance = torch.zeros(len(joints_name), 1)
    for node_index in range(len(joints_name)):
        distance = 0
        current_index = node_index
        while current_index != -1:
            origin = offset[current_index]
            distance += calc_distance(origin[0], origin[1], origin[2])
            current_index = parent[current_index]
        root_distance[node_index] = distance
    return root_distance


def calc_shoulder_distance(parent, joints_name, offset, cfg) -> torch.Tensor:
    # distance to shoulder joint
    shoulder_distance = torch.zeros(len(joints_name), 1)
    for node_index in range(len(joints_name)):
        distance = 0
        current_index = node_index
        while (
            current_index != -1 and joints_name[current_index] not in cfg["shoulders"]
        ):
            origin = offset[current_index]
            distance += calc_distance(origin[0], origin[1], origin[2])
            current_index = parent[current_index]
        shoulder_distance[node_index] = distance
    return shoulder_distance


def calc_elbow_distance(parent, joints_name, offset, cfg) -> torch.Tensor:
    elbow_distance = torch.zeros(len(joints_name), 1)
    for node_index in range(len(joints_name)):
        distance = 0
        current_index = node_index
        while current_index != -1 and joints_name[current_index] not in cfg["elbows"]:
            origin = offset[current_index]
            distance += calc_distance(origin[0], origin[1], origin[2])
            current_index = parent[current_index]
        elbow_distance[node_index] = distance
    return elbow_distance


def get_joints_rotation_axis(joints, joints_name, cfg):
    return torch.stack(
        [
            torch.Tensor(joints[joint]["axis"])
            if joint != cfg["root_name"]
            else torch.zeros(3)
            for joint in joints_name
        ],
        dim=0,
    )


def get_joints_lim(joints, joints_name, cfg):
    # joint limit
    lower = [
        torch.Tensor([joints[joint]["lower"]])
        if joint != cfg["root_name"]
        else torch.zeros(1)
        for joint in joints_name
    ]
    lower = torch.stack(lower, dim=0)
    upper = [
        torch.Tensor([joints[joint]["upper"]])
        if joint != cfg["root_name"]
        else torch.zeros(1)
        for joint in joints_name
    ]
    upper = torch.stack(upper, dim=0)
    return upper, lower


def yumi2graph(urdf_file: str, cfg: dict) -> Data:
    """
    convert Yumi URDF to graph
    urdf_file: robot urdf file path
    cfg: the way joints construct graph
    """
    # load joints
    robot, joints = load_joints_from_urdf(urdf_file)
    joints_name = cfg["joints_name"]
    joints_index = {name: i for i, name in enumerate(joints_name)}
    # collect edge data
    edge_index, edge_attr = collect_edge_data_from_joints(joints, joints_index, cfg)
    # number of nodes
    num_nodes = len(joints_name)
    # end effector mask
    ee_mask = torch.zeros(len(joints_name), 1).bool()
    for ee in cfg["end_effectors"]:
        ee_mask[joints_index[ee]] = True
    # shoulder mask
    sh_mask = torch.zeros(len(joints_name), 1).bool()
    for sh in cfg["shoulders"]:
        sh_mask[joints_index[sh]] = True
    # elbow mask
    el_mask = torch.zeros(len(joints_name), 1).bool()
    for el in cfg["elbows"]:
        el_mask[joints_index[el]] = True
    # node parent
    parent = -torch.ones(len(joints_name)).long()
    for edge in edge_index.permute(1, 0):
        parent[edge[1]] = edge[0]
    # node offset
    offset = torch.stack(
        [torch.Tensor(joints[joint]["origin"]) for joint in joints_name], dim=0
    )
    # change root offset to store init pose
    init_pose = {}
    fk = robot.link_fk()
    for link, matrix in fk.items():
        init_pose[link.name] = matrix_to_xyz_rpy(matrix)
    origin = torch.zeros(6)
    for root in cfg["root_name"]:
        offset[joints_index[root]] = torch.Tensor(init_pose[joints[root]["child"]])
        origin[:3] += offset[joints_index[root]][:3]
    origin /= 2
    # move root joints' center to origin
    for root in cfg["root_name"]:
        offset[joints_index[root]] -= origin
    # calculate distances
    root_distance = calc_root_distance(parent, joints_name, offset)
    # distance to shoulder
    shoulder_distance = calc_shoulder_distance(parent, joints_name, offset, cfg)
    # distance to elbow
    elbow_distance = calc_elbow_distance(parent, joints_name, offset, cfg)
    # rotation axis
    axis = get_joints_rotation_axis(joints, joints_name, cfg)
    # joint limit
    upper, lower = get_joints_lim(joints, joints_name, cfg)
    # skeleton
    data = Data(
        x=torch.zeros(num_nodes, 1),
        edge_index=edge_index,
        edge_attr=edge_attr,
        skeleton_type=0,
        topology_type=0,
        ee_mask=ee_mask,
        sh_mask=sh_mask,
        el_mask=el_mask,
        root_dist=root_distance,
        shoulder_dist=shoulder_distance,
        elbow_dist=elbow_distance,
        num_nodes=num_nodes,
        parent=parent,
        offset=offset,
        axis=axis,
        lower=lower,
        upper=upper,
    )
    return data


def hand2graph(urdf_file, cfg):
    """
    convert Inspire Hand URDF graph
    """
    # load URDF
    robot, joints = load_joints_from_urdf(urdf_file)
    joints_name = cfg["joints_name"]
    joints_index = {name: i for i, name in enumerate(joints_name)}
    # collect edge data
    edge_index, edge_attr = collect_edge_data_from_joints(joints, joints_index, cfg)
    # number of nodes
    num_nodes = len(joints_name)
    # end effector mask
    ee_mask = torch.zeros(len(joints_name), 1).bool()
    for ee in cfg["end_effectors"]:
        ee_mask[joints_index[ee]] = True
    # elbow mask
    el_mask = torch.zeros(len(joints_name), 1).bool()
    for el in cfg["elbows"]:
        el_mask[joints_index[el]] = True
    # node parent
    parent = -torch.ones(len(joints_name)).long()
    for edge in edge_index.permute(1, 0):
        parent[edge[1]] = edge[0]
    # node offset
    offset = []
    for joint in joints_name:
        offset.append(torch.Tensor(joints[joint]["origin"]))
    offset = torch.stack(offset, dim=0)
    # calculate distances
    root_distance = calc_root_distance(parent, joints_name, offset)
    # distance to elbow
    elbow_distance = calc_elbow_distance(parent, joints_name, offset, cfg)
    # rotation axis
    axis = get_joints_rotation_axis(joints, joints_name, cfg)
    # joint limit
    upper, lower = get_joints_lim(joints, joints_name, cfg)
    # skeleton
    data = Data(
        x=torch.zeros(num_nodes, 1),
        edge_index=edge_index,
        edge_attr=edge_attr,
        skeleton_type=0,
        topology_type=0,
        ee_mask=ee_mask,
        el_mask=el_mask,
        root_dist=root_distance,
        elbow_dist=elbow_distance,
        num_nodes=num_nodes,
        parent=parent,
        offset=offset,
        axis=axis,
        lower=lower,
        upper=upper,
    )
    # data for arm with hand
    data.hand_x = data.x
    data.hand_edge_index = data.edge_index
    data.hand_edge_attr = data.edge_attr
    data.hand_ee_mask = data.ee_mask
    data.hand_el_mask = data.el_mask
    data.hand_root_dist = data.root_dist
    data.hand_elbow_dist = data.elbow_dist
    data.hand_num_nodes = data.num_nodes
    data.hand_parent = data.parent
    data.hand_offset = data.offset
    data.hand_axis = data.axis
    data.hand_lower = data.lower
    data.hand_upper = data.upper
    return data


if __name__ == "__main__":
    # yumi graph
    graph = yumi2graph(urdf_file="../data/target/yumi/yumi.urdf", cfg=yumi_cfg)
    print("yumi", graph)
    # hand graph
    graph = hand2graph(
        urdf_file="../data/target/yumi-with-hands/yumi-with-hands.urdf", cfg=hand_cfg
    )
    print("hand", graph)
