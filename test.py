import torch
from torch_geometric.data import Batch
from models.loss import calculate_all_loss, Loss
import time


def test_epoch(
    model,
    loss_criterion,
    dataloader,
    target_skeleton,
    epoch,
    logger,
    writer,
    device,
    loss_gain=None,
):
    logger.info("Testing Epoch {}".format(epoch + 1).center(100, "-"))
    start_time = time.time()

    model.eval()
    losses = Loss()
    with torch.no_grad():
        for batch_idx, data_list in enumerate(dataloader):
            for target_idx, target in enumerate(target_skeleton):
                # fetch target
                target_list = [target for data in data_list]
                # forward
                (
                    z,
                    target_ang,
                    target_pos,
                    target_rot,
                    target_global_pos,
                    l_hand_ang,
                    l_hand_pos,
                    r_hand_ang,
                    r_hand_pos,
                ) = model(
                    Batch.from_data_list(data_list).to(device),
                    Batch.from_data_list(target_list).to(device),
                )
                # calculate all loss
                (_, losses) = calculate_all_loss(
                    data_list,
                    target_list,
                    loss_criterion,
                    z,
                    target_ang,
                    target_pos,
                    target_rot,
                    target_global_pos,
                    l_hand_pos,
                    r_hand_pos,
                    losses,
                    loss_gain,
                )
    # Compute average loss
    test_loss = sum(losses.sum) / len(losses.sum)
    ee_loss = sum(losses.ee) / len(losses.ee)
    vec_loss = sum(losses.vec) / len(losses.vec)
    col_loss = sum(losses.col) / len(losses.col)
    lim_loss = sum(losses.lim) / len(losses.lim)
    ori_loss = sum(losses.ori) / len(losses.ori)
    fin_loss = sum(losses.fin) / len(losses.fin)
    reg_loss = sum(losses.reg) / len(losses.reg)
    # Log
    writer.add_scalars("testing_loss", {"test": test_loss}, epoch + 1)
    writer.add_scalars("end_effector_loss", {"test": ee_loss}, epoch + 1)
    writer.add_scalars("vector_loss", {"test": vec_loss}, epoch + 1)
    writer.add_scalars("collision_loss", {"test": col_loss}, epoch + 1)
    writer.add_scalars("joint_limit_loss", {"test": lim_loss}, epoch + 1)
    writer.add_scalars("orientation_loss", {"test": ori_loss}, epoch + 1)
    writer.add_scalars("finger_loss", {"test": fin_loss}, epoch + 1)
    writer.add_scalars("regularization_loss", {"test": reg_loss}, epoch + 1)
    end_time = time.time()
    logger.info(
        "Epoch {:03d} | Testing Time {:.2f} s | Avg Testing {:.2f} | Avg EE {:.2f} | "
        "Avg Vec {:.2f} | Avg Col {:.2f} | Avg Lim {:.2f} | Avg Ori {:.2f} | "
        "Avg Fin {:.2f} | Avg Reg {:.2f}".format(
            epoch + 1,
            end_time - start_time,
            test_loss,
            ee_loss,
            vec_loss,
            col_loss,
            lim_loss,
            ori_loss,
            fin_loss,
            reg_loss,
        )
    )
    return test_loss
