import inspect
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

from models.kinematics import ForwardKinematicsURDF, ForwardKinematicsAxis

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, current_dir)


class SpatialBasicBlock(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        edge_channels,
        aggr="add",
        batch_norm=False,
        bias=True,
        **kwargs
    ):
        super(SpatialBasicBlock, self).__init__(aggr=aggr, **kwargs)
        self.batch_norm = batch_norm
        # network architecture
        self.lin = nn.Linear(2 * in_channels + edge_channels, out_channels, bias=bias)
        self.upsample = nn.Linear(in_channels, out_channels, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.bn(out) if self.batch_norm else out
        out += self.upsample(x[1])
        return out

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return F.leaky_relu(self.lin(z))


class Encoder(torch.nn.Module):
    def __init__(self, channels, dim):
        super(Encoder, self).__init__()
        self.conv1 = SpatialBasicBlock(
            in_channels=channels, out_channels=16, edge_channels=dim
        )
        self.conv2 = SpatialBasicBlock(
            in_channels=16, out_channels=32, edge_channels=dim
        )
        self.conv3 = SpatialBasicBlock(
            in_channels=32, out_channels=64, edge_channels=dim
        )

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Keyword arguments:
        x -- joint angles [num_nodes, num_node_features]
        edge_index -- edge index [2, num_edges]
        edge_attr -- edge features [num_edges, num_edge_features]
        """
        out = self.conv1(x, edge_index, edge_attr)
        out = self.conv2(out, edge_index, edge_attr)
        out = self.conv3(out, edge_index, edge_attr)
        return out


class Decoder(torch.nn.Module):
    def __init__(self, channels, dim):
        super(Decoder, self).__init__()
        self.conv1 = SpatialBasicBlock(
            in_channels=64 + 2, out_channels=32, edge_channels=dim
        )
        self.conv2 = SpatialBasicBlock(
            in_channels=32, out_channels=16, edge_channels=dim
        )
        self.conv3 = SpatialBasicBlock(
            in_channels=16, out_channels=channels, edge_channels=dim
        )

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, x, edge_index, edge_attr, lower, upper):
        """
        Keyword arguments:
        x -- joint angles [num_nodes, num_node_features]
        edge_index -- edge index [2, num_edges]
        edge_attr -- edge features [num_edges, num_edge_features]
        """
        x = torch.cat([x, lower, upper], dim=1)
        out = self.conv1(x, edge_index, edge_attr)
        out = self.conv2(out, edge_index, edge_attr)
        out = self.conv3(out, edge_index, edge_attr).tanh()
        return out


class Return_Arm_Data:
    def __init__(self, arm_ang=None, arm_pos=None, arm_rot=None, arm_global_pos=None):
        self.arm_ang = arm_ang
        self.arm_pos = arm_pos
        self.arm_rot = arm_rot
        self.arm_global_pos = arm_global_pos


class Return_Hand_Data:
    def __init__(
        self, l_hand_ang=None, l_hand_pos=None, r_hand_ang=None, r_hand_pos=None
    ):
        self.l_hand_ang = l_hand_ang
        self.l_hand_pos = l_hand_pos
        self.r_hand_ang = r_hand_ang
        self.r_hand_pos = r_hand_pos


class ArmNet(torch.nn.Module):
    def __init__(self):
        super(ArmNet, self).__init__()
        self.encoder = Encoder(6, 3)
        self.transform = nn.Sequential(nn.Linear(6 * 64, 14 * 64), nn.Tanh())
        self.decoder = Decoder(1, 6)
        self.fk = ForwardKinematicsURDF()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.transform = nn.Sequential(nn.Linear(6 * 64, 14 * 64), nn.Tanh())
        self.decoder.reset_parameters()

    def forward(self, data, target):
        return self.decode(self.encode(data), target)

    def encode(self, data):
        z = self.encoder(data.x, data.edge_index, data.edge_attr)
        num_graphs = data.num_graphs
        z = (
            self.transform(z.view(num_graphs, -1, 64).view(num_graphs, -1))
            .view(num_graphs, -1, 64)
            .view(-1, 64)
        )
        return z

    def decode(self, z, target):
        lower, upper = target.lower, target.upper
        # Decoder outputs joints' angles
        ang = self.decoder(z, target.edge_index, target.edge_attr, lower, upper)
        ang = lower + (upper - lower) * (ang + 1) / 2
        pos, rot, global_pos = self.fk(
            ang, target.parent, target.offset, target.num_graphs
        )
        return Return_Arm_Data(ang, pos, rot, global_pos)


class HandNet(torch.nn.Module):
    def __init__(self):
        super(HandNet, self).__init__()
        self.encoder = Encoder(3, 3)
        self.transform = nn.Sequential(nn.Linear(17 * 64, 13 * 64), nn.Tanh())
        self.decoder = Decoder(1, 6)
        self.fk = ForwardKinematicsAxis()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.transform = nn.Sequential(nn.Linear(17 * 64, 13 * 64), nn.Tanh())
        self.decoder.reset_parameters()

    def forward(self, data, target):
        return self.decode(self.encode(data), target)

    def encode(self, data):
        num_graphs = data.num_graphs
        x = torch.cat([data.l_hand_x, data.r_hand_x], dim=0)
        edge_index = torch.cat(
            [data.l_hand_edge_index, data.r_hand_edge_index + data.l_hand_x.size(0)],
            dim=1,
        )
        edge_attr = torch.cat([data.l_hand_edge_attr, data.r_hand_edge_attr], dim=0)
        z = self.encoder(x, edge_index, edge_attr)
        z = (
            self.transform(z.view(2 * num_graphs, -1, 64).view(2 * num_graphs, -1))
            .view(2 * num_graphs, -1, 64)
            .view(-1, 64)
        )
        return z

    def decode(self, z: torch.Tensor, target):
        hand_edge_index = target.hand_edge_index
        edge_index = torch.cat(
            [hand_edge_index, hand_edge_index + z.size(0) // 2], dim=1
        )
        edge_attr = torch.cat([target.hand_edge_attr, target.hand_edge_attr], dim=0)
        lower = torch.cat([target.hand_lower, target.hand_lower], dim=0)
        upper = torch.cat([target.hand_upper, target.hand_upper], dim=0)
        offset = torch.cat([target.hand_offset, target.hand_offset], dim=0)
        parent = torch.cat([target.hand_parent, target.hand_parent], dim=0)
        num_graphs = 2 * target.num_graphs
        axis = torch.cat([target.hand_axis, target.hand_axis], dim=0)

        hand_ang = self.decoder(z, edge_index, edge_attr, lower, upper)
        hand_ang = lower + (upper - lower) * (hand_ang + 1) / 2
        hand_pos, _, _ = self.fk(hand_ang, parent, offset, num_graphs, axis)

        half = hand_ang.size(0) // 2
        l_hand_ang, r_hand_ang = hand_ang[:half, :], hand_ang[half:, :]
        l_hand_pos, r_hand_pos = hand_pos[:half, :], hand_pos[half:, :]
        return Return_Hand_Data(l_hand_ang, l_hand_pos, r_hand_ang, r_hand_pos)


class YumiNet(torch.nn.Module):
    def __init__(self):
        super(YumiNet, self).__init__()
        self.arm_net = ArmNet()
        self.hand_net = HandNet()

    def reset_parameters(self):
        self.arm_net.reset_parameters()
        self.hand_net.reset_parameters()

    def forward(self, data, target):
        return self.decode(self.encode(data), target)

    def encode(self, data):
        arm_z = self.arm_net.encode(data)
        hand_z = self.hand_net.encode(data)
        z = torch.cat([arm_z, hand_z], dim=0)
        return z

    def decode(self, z: torch.Tensor, target):
        half = target.num_nodes
        arm_z, hand_z = z[:half, :], z[half:, :]
        arm_data = self.arm_net.decode(arm_z, target)
        hand_data = self.hand_net.decode(hand_z, target)
        return z, arm_data, hand_data
