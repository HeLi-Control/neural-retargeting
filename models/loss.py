import torch
import torch.nn as nn
from kornia.geometry.conversions import quaternion_to_rotation_matrix
from models.model import Return_Hand_Data, Return_Arm_Data


class Loss:
    def __init__(
        self,
        ee=None,
        vec=None,
        col=None,
        lim=None,
        ori=None,
        fin=None,
        reg=None,
        sum=None,
    ):
        self.ee = ee
        self.vec = vec
        self.col = col
        self.lim = lim
        self.ori = ori
        self.fin = fin
        self.reg = reg
        self.sum = sum


def calculate_all_loss(
    data_list,
    target_list,
    loss_criterion: Loss,
    z,
    target_data: Return_Arm_Data,
    hand_data: Return_Hand_Data,
    losses: Loss,
    loss_gain=None,
) -> (torch.Tensor, Loss):
    """
    Calculate All Loss
    """
    losses = Loss(
        ee=losses.ee if losses.ee is not None else [],
        vec=losses.vec if losses.vec is not None else [],
        col=losses.col if losses.col is not None else [],
        lim=losses.lim if losses.lim is not None else [],
        ori=losses.ori if losses.ori is not None else [],
        fin=losses.fin if losses.fin is not None else [],
        reg=losses.reg if losses.reg is not None else [],
        sum=losses.sum if losses.sum is not None else [],
    )
    loss_gain = loss_gain if loss_gain is not None else torch.ones(7).unsqueeze(dim=1)

    target_pos = target_data.target_pos
    target_global_pos = target_data.target_global_pos
    target_rot = target_data.target_rot
    target_ang = target_data.target_ang
    l_hand_pos = hand_data.l_hand_pos
    r_hand_pos = hand_data.r_hand_pos

    # End effector loss
    ee_loss = (
        calculate_ee_loss(data_list, target_list, target_pos, loss_criterion.ee)
        * float(loss_gain[0])
        if loss_criterion.ee
        else torch.tensor([0])
    )
    losses.ee.append(ee_loss.item())
    # Vector loss
    vec_loss = (
        calculate_vec_loss(data_list, target_list, target_pos, loss_criterion.vec)
        * float(loss_gain[1])
        if loss_criterion.vec
        else torch.tensor([0])
    )
    losses.vec.append(vec_loss.item())
    # Collision loss
    collision_loss = (
        loss_criterion.col(
            target_global_pos.view(len(target_list), -1, 3),
            target_list[0].edge_index,
            target_rot.view(len(target_list), -1, 9),
            target_list[0].ee_mask,
        )
        * float(loss_gain[2])
        if loss_criterion.col
        else torch.tensor([0])
    )
    losses.col.append(collision_loss.item())
    # joint limit loss
    lim_loss = (
        calculate_lim_loss(target_list, target_ang, loss_criterion.lim)
        * float(loss_gain[3])
        if loss_criterion.lim
        else torch.tensor([0])
    )
    losses.lim.append(lim_loss.item())
    # end effector orientation loss
    ori_loss = (
        calculate_ori_loss(data_list, target_list, target_rot, loss_criterion.ori)
        * float(loss_gain[4])
        if loss_criterion.ori
        else torch.tensor([0])
    )
    losses.ori.append(ori_loss.item())
    # finger similarity loss
    fin_loss = (
        calculate_fin_loss(
            data_list, target_list, l_hand_pos, r_hand_pos, loss_criterion.fin
        )
        * float(loss_gain[5])
        if loss_criterion.fin
        else torch.tensor([0])
    )
    losses.fin.append(fin_loss.item())
    # regularization loss
    reg_loss = (
        loss_criterion.reg(z.view(len(target_list), -1, 64)) * float(loss_gain[6])
        if loss_criterion.reg
        else torch.tensor([0])
    )
    losses.reg.append(reg_loss.item())
    # total loss
    loss = (
        ee_loss + vec_loss + collision_loss + lim_loss + ori_loss + fin_loss + reg_loss
    )
    losses.sum.append(loss.item())
    return loss, losses


def calculate_ee_loss(data_list, target_list, target_pos, criterion) -> torch.Tensor:
    """
    Calculate End Effector Position Loss
    """
    device = target_pos.device
    # mask for joints unnecessary to calculate
    target_mask = torch.cat([data.ee_mask for data in target_list]).to(device)
    source_mask = torch.cat([data.ee_mask for data in data_list]).to(device)
    # calculate end effector relative position
    target_ee = torch.masked_select(target_pos, target_mask).view(-1, 3)
    source_pos = torch.cat([data.pos for data in data_list]).to(device)
    source_ee = torch.masked_select(source_pos, source_mask).view(-1, 3)
    # normalize
    target_root_dist = torch.cat([data.root_dist for data in target_list]).to(device)
    source_root_dist = torch.cat([data.root_dist for data in data_list]).to(device)
    target_ee /= torch.masked_select(target_root_dist, target_mask).unsqueeze(1)
    source_ee /= torch.masked_select(source_root_dist, source_mask).unsqueeze(1)
    ee_loss = criterion(target_ee, source_ee)
    return ee_loss


def calculate_vec_loss(data_list, target_list, target_pos, criterion) -> torch.Tensor:
    """
    Calculate Vector Loss
    """
    device = target_pos.device
    # get masks
    target_el_mask = torch.cat([data.el_mask for data in target_list]).to(device)
    target_ee_mask = torch.cat([data.ee_mask for data in target_list]).to(device)
    source_el_mask = torch.cat([data.el_mask for data in data_list]).to(device)
    source_ee_mask = torch.cat([data.ee_mask for data in data_list]).to(device)
    # get the desired joints' position
    target_el = torch.masked_select(target_pos, target_el_mask).view(-1, 3)
    target_ee = torch.masked_select(target_pos, target_ee_mask).view(-1, 3)
    source_pos = torch.cat([data.pos for data in data_list]).to(device)
    source_el = torch.masked_select(source_pos, source_el_mask).view(-1, 3)
    source_ee = torch.masked_select(source_pos, source_ee_mask).view(-1, 3)
    # calculate elbow to end effector vector
    target_vector = target_ee - target_el
    source_vector = source_ee - source_el
    # normalize
    target_elbow_dist = (
        torch.cat([data.elbow_dist for data in target_list]).to(device) / 2
    )
    source_elbow_dist = (
        torch.cat([data.elbow_dist for data in data_list]).to(device) / 2
    )
    target_vector /= torch.masked_select(target_elbow_dist, target_ee_mask).unsqueeze(1)
    source_vector /= torch.masked_select(source_elbow_dist, source_ee_mask).unsqueeze(1)
    vec_loss = criterion(target_vector, source_vector)
    return vec_loss


def calculate_lim_loss(target_list, target_ang, criterion) -> torch.Tensor:
    """
    Calculate Joint Limit Loss
    """
    target_lower = torch.cat([data.lower for data in target_list]).to(target_ang.device)
    target_upper = torch.cat([data.upper for data in target_list]).to(target_ang.device)
    lim_loss = criterion(target_ang, target_lower, target_upper)
    return lim_loss


def calculate_ori_loss(data_list, target_list, target_rot, criterion) -> torch.Tensor:
    """
    Calculate Orientation Loss
    """
    device = target_rot.device
    # get masks
    target_mask = torch.cat([data.ee_mask for data in target_list]).to(device)
    source_mask = torch.cat([data.ee_mask for data in data_list]).to(device)
    # calculate the rotation matrix vector
    target_rot = target_rot.view(-1, 9)
    source_quaternion = torch.cat([data.q for data in data_list]).to(device)
    source_rot = quaternion_to_rotation_matrix(source_quaternion).view(-1, 9)
    target_ori = torch.masked_select(target_rot, target_mask)
    source_ori = torch.masked_select(source_rot, source_mask)
    ori_loss = criterion(target_ori, source_ori)
    return ori_loss


def calculate_fin_loss(
    data_list, target_list, l_hand_pos, r_hand_pos, criterion
) -> torch.Tensor:
    """
    Calculate Finger Similarity Loss
    """
    # left hand
    device = l_hand_pos.device
    # get masks
    target_el_mask = torch.cat([data.hand_el_mask for data in target_list]).to(device)
    target_ee_mask = torch.cat([data.hand_ee_mask for data in target_list]).to(device)
    source_el_mask = torch.cat([data.l_hand_el_mask for data in data_list]).to(device)
    source_ee_mask = torch.cat([data.l_hand_ee_mask for data in data_list]).to(device)
    # calculate finger elbow to end effector vector
    target_el = torch.masked_select(l_hand_pos, target_el_mask).view(-1, 3)
    target_ee = torch.masked_select(l_hand_pos, target_ee_mask).view(-1, 3)
    source_l_hand_pos = torch.cat([data.l_hand_pos for data in data_list]).to(device)
    source_el = torch.masked_select(source_l_hand_pos, source_el_mask).view(-1, 3)
    source_ee = torch.masked_select(source_l_hand_pos, source_ee_mask).view(-1, 3)
    target_vector = target_ee - target_el
    source_vector = source_ee - source_el
    # normalize
    target_elbow_dist = torch.cat([data.hand_elbow_dist for data in target_list]).to(
        device
    )
    source_elbow_dist = torch.cat([data.l_hand_elbow_dist for data in data_list]).to(
        device
    )
    target_vector /= torch.masked_select(target_elbow_dist, target_ee_mask).unsqueeze(1)
    source_vector /= torch.masked_select(source_elbow_dist, source_ee_mask).unsqueeze(1)
    l_fin_loss = criterion(target_vector, source_vector)

    # right hand
    device = r_hand_pos.device
    # get masks
    target_el_mask = torch.cat([data.hand_el_mask for data in target_list]).to(device)
    target_ee_mask = torch.cat([data.hand_ee_mask for data in target_list]).to(device)
    source_el_mask = torch.cat([data.r_hand_el_mask for data in data_list]).to(device)
    source_ee_mask = torch.cat([data.r_hand_ee_mask for data in data_list]).to(device)
    # calculate finger elbow to end effector vector
    target_el = torch.masked_select(r_hand_pos, target_el_mask).view(-1, 3)
    target_ee = torch.masked_select(r_hand_pos, target_ee_mask).view(-1, 3)
    source_r_hand_pos = torch.cat([data.r_hand_pos for data in data_list]).to(device)
    source_el = torch.masked_select(source_r_hand_pos, source_el_mask).view(-1, 3)
    source_ee = torch.masked_select(source_r_hand_pos, source_ee_mask).view(-1, 3)
    target_vector = target_ee - target_el
    source_vector = source_ee - source_el
    # normalize
    target_elbow_dist = torch.cat([data.hand_elbow_dist for data in target_list]).to(
        device
    )
    source_elbow_dist = torch.cat([data.r_hand_elbow_dist for data in data_list]).to(
        device
    )
    target_vector /= torch.masked_select(target_elbow_dist, target_ee_mask).unsqueeze(1)
    source_vector /= torch.masked_select(source_elbow_dist, source_ee_mask).unsqueeze(1)
    r_fin_loss = criterion(target_vector, source_vector)

    # average loss
    fin_loss = (l_fin_loss + r_fin_loss) / 2
    return fin_loss


def sphere_capsule_dist_square(
    sphere, capsule_p0, capsule_p1, batch_size, num_nodes, num_edges
):
    # condition 1: p0 is the closest point
    vec_p0_p1 = capsule_p1 - capsule_p0  # vector p0-p1 [batch_size, num_edges//2, 3]
    vec_p0_pr = sphere.unsqueeze(2).expand(
        batch_size, num_nodes // 2, num_edges // 2, 3
    ) - capsule_p0.unsqueeze(1).expand(
        batch_size, num_nodes // 2, num_edges // 2, 3
    )  # vector p0-pr [batch_size, num_nodes//2, num_edges//2, 3]
    vec_mul_p0 = torch.mul(
        vec_p0_p1.unsqueeze(1).expand(batch_size, num_nodes // 2, num_edges // 2, 3),
        vec_p0_pr,
    ).sum(
        dim=-1
    )  # vector p0-p1 * vector p0-pr [batch_size, num_nodes//2, num_edges//2]
    dist_square_p0 = torch.masked_select(vec_p0_pr.norm(dim=-1) ** 2, vec_mul_p0 <= 0)
    # print(dist_square_p0.shape)

    # condition 2: p1 is the closest point
    vec_p1_p0 = capsule_p0 - capsule_p1  # vector p1-p0 [batch_size, num_edges//2, 3]
    vec_p1_pr = sphere.unsqueeze(2).expand(
        batch_size, num_nodes // 2, num_edges // 2, 3
    ) - capsule_p1.unsqueeze(1).expand(
        batch_size, num_nodes // 2, num_edges // 2, 3
    )  # vector p1-pr [batch_size, num_nodes//2, num_edges//2, 3]
    vec_mul_p1 = torch.mul(
        vec_p1_p0.unsqueeze(1).expand(batch_size, num_nodes // 2, num_edges // 2, 3),
        vec_p1_pr,
    ).sum(
        dim=-1
    )  # vector p1-p0 * vector p1-pr [batch_size, num_nodes//2, num_edges//2]
    dist_square_p1 = torch.masked_select(vec_p1_pr.norm(dim=-1) ** 2, vec_mul_p1 <= 0)
    # print(dist_square_p1.shape)

    # condition 3: closest point in p0-p1 segment
    d = vec_mul_p0 / vec_p0_p1.norm(dim=-1).unsqueeze(1).expand(
        batch_size, num_nodes // 2, num_edges // 2
    )  # vector p0-p1 * vector p0-pr / |vector p0-p1| [batch_size, num_nodes//2, num_edges//2]
    dist_square_middle = (
        vec_p0_pr.norm(dim=-1) ** 2 - d**2
    )  # distance square [batch_size, num_nodes//2, num_edges//2]
    dist_square_middle = torch.masked_select(
        dist_square_middle, (vec_mul_p0 > 0) & (vec_mul_p1 > 0)
    )
    # print(dist_square_middle.shape)

    return torch.cat([dist_square_p0, dist_square_p1, dist_square_middle])


class CollisionLoss(nn.Module):
    """
    Collision Loss
    """

    def __init__(self, threshold, mode="capsule-capsule"):
        super(CollisionLoss, self).__init__()
        self.threshold = threshold
        self.mode = mode

    def forward(self, pos, edge_index, rot, ee_mask):
        """
        Keyword arguments:
        pos -- joint positions [batch_size, num_nodes, 3]
        edge_index -- edge index [2, num_edges]
        """
        batch_size = pos.shape[0]
        num_nodes = pos.shape[1]
        num_edges = edge_index.shape[1]
        loss = None

        # sphere-sphere detection
        if self.mode == "sphere-sphere":
            l_sphere = pos[:, : num_nodes // 2, :]
            r_sphere = pos[:, num_nodes // 2 :, :]
            l_sphere = l_sphere.unsqueeze(1).expand(
                batch_size, num_nodes // 2, num_nodes // 2, 3
            )
            r_sphere = r_sphere.unsqueeze(2).expand(
                batch_size, num_nodes // 2, num_nodes // 2, 3
            )
            dist_square = torch.sum(torch.pow(l_sphere - r_sphere, 2), dim=-1)
            mask = torch.tensor((dist_square < self.threshold**2) & (dist_square > 0))
            loss = (
                torch.sum(torch.exp(-1 * torch.masked_select(dist_square, mask)))
                / batch_size
            )
        # sphere-capsule detection
        if self.mode == "sphere-capsule":
            # capsule p0 & p1
            p0 = pos[:, edge_index[0], :]
            p1 = pos[:, edge_index[1], :]
            # print(edge_index.shape, p0.shape, p1.shape)

            # left sphere & right capsule
            l_sphere = pos[:, : num_nodes // 2, :]
            r_capsule_p0 = p0[:, num_edges // 2 :, :]
            r_capsule_p1 = p1[:, num_edges // 2 :, :]
            dist_square_1 = sphere_capsule_dist_square(
                l_sphere, r_capsule_p0, r_capsule_p1, batch_size, num_nodes, num_edges
            )

            # left capsule & right sphere
            r_sphere = pos[:, num_nodes // 2 :, :]
            l_capsule_p0 = p0[:, : num_edges // 2, :]
            l_capsule_p1 = p1[:, : num_edges // 2, :]
            dist_square_2 = sphere_capsule_dist_square(
                r_sphere, l_capsule_p0, l_capsule_p1, batch_size, num_nodes, num_edges
            )

            # calculate loss
            dist_square = torch.cat([dist_square_1, dist_square_2])
            mask = torch.tensor((dist_square < self.threshold**2) & (dist_square > 0))
            loss = (
                torch.sum(torch.exp(-1 * torch.masked_select(dist_square, mask)))
                / batch_size
            )

        # capsule-capsule detection
        if self.mode == "capsule-capsule":
            # capsule p0 & p1
            p0 = pos[:, edge_index[0], :]
            p1 = pos[:, edge_index[1], :]
            # left capsule
            l_capsule_p0 = p0[:, : num_edges // 2, :]
            l_capsule_p1 = p1[:, : num_edges // 2, :]
            # right capsule
            r_capsule_p0 = p0[:, num_edges // 2 :, :]
            r_capsule_p1 = p1[:, num_edges // 2 :, :]
            # add capsule for left hand & right hand(for yumi)
            ee_pos = torch.masked_select(pos, ee_mask.to(pos.device)).view(-1, 3)
            ee_rot = torch.masked_select(rot, ee_mask.to(pos.device)).view(-1, 3, 3)
            offset = (
                torch.Tensor([[[0], [0], [0.2]]])
                .repeat(ee_rot.size(0), 1, 1)
                .to(pos.device)
            )
            hand_pos = torch.bmm(ee_rot, offset).squeeze() + ee_pos
            l_ee_pos = ee_pos[::2, :].unsqueeze(1)
            l_hand_pos = hand_pos[::2, :].unsqueeze(1)
            r_ee_pos = ee_pos[1::2, :].unsqueeze(1)
            r_hand_pos = hand_pos[1::2, :].unsqueeze(1)
            l_capsule_p0 = torch.cat([l_capsule_p0, l_ee_pos], dim=1)
            l_capsule_p1 = torch.cat([l_capsule_p1, l_hand_pos], dim=1)
            r_capsule_p0 = torch.cat([r_capsule_p0, r_ee_pos], dim=1)
            r_capsule_p1 = torch.cat([r_capsule_p1, r_hand_pos], dim=1)
            num_edges += 2
            # print(l_capsule_p0.shape, l_capsule_p1.shape, r_capsule_p0.shape, r_capsule_p1.shape)
            # calculate loss
            dist_square = self.capsule_capsule_dist_square(
                l_capsule_p0,
                l_capsule_p1,
                r_capsule_p0,
                r_capsule_p1,
                batch_size,
                num_edges,
            )
            mask = (dist_square < 0.1**2) & (dist_square > 0)
            mask[:, 6, 6] = (dist_square[:, 6, 6] > self.threshold**2) & (
                dist_square[:, 6, 6] > 0
            )
            loss = (
                torch.sum(torch.exp(-1 * torch.masked_select(dist_square, mask)))
                / batch_size
            )
        return loss

    def capsule_capsule_dist_square(
        self, capsule_p0, capsule_p1, capsule_q0, capsule_q1, batch_size, num_edges
    ):
        # expand left capsule
        capsule_p0 = capsule_p0.unsqueeze(1).expand(
            batch_size, num_edges // 2, num_edges // 2, 3
        )
        capsule_p1 = capsule_p1.unsqueeze(1).expand(
            batch_size, num_edges // 2, num_edges // 2, 3
        )
        # expand right capsule
        capsule_q0 = capsule_q0.unsqueeze(2).expand(
            batch_size, num_edges // 2, num_edges // 2, 3
        )
        capsule_q1 = capsule_q1.unsqueeze(2).expand(
            batch_size, num_edges // 2, num_edges // 2, 3
        )
        # basic variables
        a = torch.mul(capsule_p1 - capsule_p0, capsule_p1 - capsule_p0).sum(dim=-1)
        b = torch.mul(capsule_p1 - capsule_p0, capsule_q1 - capsule_q0).sum(dim=-1)
        c = torch.mul(capsule_q1 - capsule_q0, capsule_q1 - capsule_q0).sum(dim=-1)
        d = torch.mul(capsule_p1 - capsule_p0, capsule_p0 - capsule_q0).sum(dim=-1)
        e = torch.mul(capsule_q1 - capsule_q0, capsule_p0 - capsule_q0).sum(dim=-1)
        # f = torch.mul(capsule_p0 - capsule_q0, capsule_p0 - capsule_q0).sum(dim=-1)
        # initialize s, t to zero
        s = torch.zeros(batch_size, num_edges // 2, num_edges // 2).to(
            capsule_p0.device
        )
        t = torch.zeros(batch_size, num_edges // 2, num_edges // 2).to(
            capsule_p0.device
        )
        one = torch.ones(batch_size, num_edges // 2, num_edges // 2).to(
            capsule_p0.device
        )
        # calculate coefficient
        det = a * c - b**2
        bte = b * e
        ctd = c * d
        ate = a * e
        btd = b * d
        # nonparallel segments
        # region 6
        s = torch.where((det > 0) & (bte <= ctd) & (e <= 0) & (-d >= a), one, s)
        s = torch.where(
            (det > 0) & (bte <= ctd) & (e <= 0) & (-d < a) & (-d > 0), -d / a, s
        )
        # region 5
        t = torch.where((det > 0) & (bte <= ctd) & (e > 0) & (e < c), e / c, t)
        # region 4
        s = torch.where(
            (det > 0) & (bte <= ctd) & (e > 0) & (e >= c) & (b - d >= a), one, s
        )
        s = torch.where(
            (det > 0) & (bte <= ctd) & (e > 0) & (e >= c) & (b - d < a) & (b - d > 0),
            (b - d) / a,
            s,
        )
        t = torch.where((det > 0) & (bte <= ctd) & (e > 0) & (e >= c), one, t)
        # region 8
        s = torch.where(
            (det > 0)
            & (bte > ctd)
            & (bte - ctd >= det)
            & (b + e <= 0)
            & (-d > 0)
            & (-d < a),
            -d / a,
            s,
        )
        s = torch.where(
            (det > 0)
            & (bte > ctd)
            & (bte - ctd >= det)
            & (b + e <= 0)
            & (-d > 0)
            & (-d >= a),
            one,
            s,
        )
        # region 1
        s = torch.where(
            (det > 0) & (bte > ctd) & (bte - ctd >= det) & (b + e > 0) & (b + e < c),
            one,
            s,
        )
        t = torch.where(
            (det > 0) & (bte > ctd) & (bte - ctd >= det) & (b + e > 0) & (b + e < c),
            (b + e) / c,
            t,
        )
        # region 2
        s = torch.where(
            (det > 0)
            & (bte > ctd)
            & (bte - ctd >= det)
            & (b + e > 0)
            & (b + e >= c)
            & (b - d > 0)
            & (b - d < a),
            (b - d) / a,
            s,
        )
        s = torch.where(
            (det > 0)
            & (bte > ctd)
            & (bte - ctd >= det)
            & (b + e > 0)
            & (b + e >= c)
            & (b - d > 0)
            & (b - d >= a),
            one,
            s,
        )
        t = torch.where(
            (det > 0) & (bte > ctd) & (bte - ctd >= det) & (b + e > 0) & (b + e >= c),
            one,
            t,
        )
        # region 7
        s = torch.where(
            (det > 0)
            & (bte > ctd)
            & (bte - ctd < det)
            & (ate <= btd)
            & (-d > 0)
            & (-d >= a),
            one,
            s,
        )
        s = torch.where(
            (det > 0)
            & (bte > ctd)
            & (bte - ctd < det)
            & (ate <= btd)
            & (-d > 0)
            & (-d < a),
            -d / a,
            s,
        )
        # region 3
        s = torch.where(
            (det > 0)
            & (bte > ctd)
            & (bte - ctd < det)
            & (ate > btd)
            & (ate - btd >= det)
            & (b - d > 0)
            & (b - d >= a),
            one,
            s,
        )
        s = torch.where(
            (det > 0)
            & (bte > ctd)
            & (bte - ctd < det)
            & (ate > btd)
            & (ate - btd >= det)
            & (b - d > 0)
            & (b - d < a),
            (b - d) / a,
            s,
        )
        t = torch.where(
            (det > 0)
            & (bte > ctd)
            & (bte - ctd < det)
            & (ate > btd)
            & (ate - btd >= det),
            one,
            t,
        )
        # region 0
        s = torch.where(
            (det > 0)
            & (bte > ctd)
            & (bte - ctd < det)
            & (ate > btd)
            & (ate - btd < det),
            (bte - ctd) / det,
            s,
        )
        t = torch.where(
            (det > 0)
            & (bte > ctd)
            & (bte - ctd < det)
            & (ate > btd)
            & (ate - btd < det),
            (ate - btd) / det,
            t,
        )
        # parallel segments
        # e <= 0
        s = torch.where((det <= 0) & (e <= 0) & (-d > 0) & (-d >= a), one, s)
        s = torch.where((det <= 0) & (e <= 0) & (-d > 0) & (-d < a), -d / a, s)
        # e >= c
        s = torch.where(
            (det <= 0) & (e > 0) & (e >= c) & (b - d > 0) & (b - d >= a), one, s
        )
        s = torch.where(
            (det <= 0) & (e > 0) & (e >= c) & (b - d > 0) & (b - d < a), (b - d) / a, s
        )
        t = torch.where((det <= 0) & (e > 0) & (e >= c), one, t)
        # 0 < e < c
        t = torch.where((det <= 0) & (e > 0) & (e < c), e / c, t)
        # print(s, t)
        s = (
            s.unsqueeze(-1)
            .expand(batch_size, num_edges // 2, num_edges // 2, 3)
            .detach()
        )
        t = (
            t.unsqueeze(-1)
            .expand(batch_size, num_edges // 2, num_edges // 2, 3)
            .detach()
        )
        w = (
            capsule_p0
            - capsule_q0
            + s * (capsule_p1 - capsule_p0)
            - t * (capsule_q1 - capsule_q0)
        )
        dist_square = torch.mul(w, w).sum(dim=-1)
        return dist_square


class JointLimitLoss(nn.Module):
    """
    Joint Limit Loss
    """

    def __init__(self):
        super(JointLimitLoss, self).__init__()

    def forward(self, ang, lower, upper):
        """
        Keyword augments:
        ang -- joint angles [batch_size*num_nodes, num_node_features]
        """
        # calculate mask with limit
        lower_mask = torch.tensor(ang < lower)
        upper_mask = torch.tensor(ang > upper)
        # calculate final loss
        lower_loss = torch.sum(torch.masked_select(lower - ang, lower_mask))
        upper_loss = torch.sum(torch.masked_select(ang - upper, upper_mask))
        loss = (lower_loss + upper_loss) / ang.shape[0]
        return loss


class RegLoss(nn.Module):
    """
    Regularization Loss
    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, z):
        # calculate final loss
        batch_size = z.shape[0]
        loss = torch.mean(torch.norm(z.view(batch_size, -1), dim=1).pow(2))
        return loss
