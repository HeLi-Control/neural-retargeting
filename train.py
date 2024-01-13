import torch
from torch_geometric.data import Batch
from models.loss import calculate_all_loss, Loss
import time


def train_epoch(
    model,
    loss_criterion,
    optimizer,
    dataloader,
    target_skeleton,
    epoch,
    logger,
    log_interval,
    writer,
    device,
    loss_gain=None,
    z_all=None,
):
    logger.info("Training Epoch {}".format(epoch + 1).center(100, "-"))
    start_time = time.time()

    model.train()
    losses = Loss()
    for batch_idx, data_list in enumerate(dataloader):
        for target_idx, target in enumerate(target_skeleton):
            # zero gradient
            optimizer.zero_grad()
            # fetch target
            target_list = [target for data in data_list]
            # forward
            if z_all is not None:
                z = z_all[batch_idx]
                _, target_data, hand_data = model.decode(
                    z, Batch.from_data_list(target_list).to(device)
                )
            else:
                z, target_data, hand_data = model(
                    Batch.from_data_list(data_list).to(device),
                    Batch.from_data_list(target_list).to(device),
                )
            # calculate all loss
            (loss, losses) = calculate_all_loss(
                data_list,
                target_list,
                loss_criterion,
                z,
                target_data,
                hand_data,
                losses,
                loss_gain,
            )
            # backward
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            # optimize
            optimizer.step()
        # log
        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                "epoch {:03d} | iteration {:03d} | Sum {:.2f} | EE {:.2f} | Vec {:.2f}"
                " | Col {:.2f} | Lim {:.2f} | Ori {:.2f} | Fin {:.2f} | Reg {:.2f}".format(
                    epoch + 1,
                    batch_idx + 1,
                    losses.sum[-1],
                    losses.ee[-1],
                    losses.vec[-1],
                    losses.col[-1],
                    losses.lim[-1],
                    losses.ori[-1],
                    losses.fin[-1],
                    losses.reg[-1],
                )
            )
    # Compute average loss
    train_loss = sum(losses.sum) / len(losses.sum)
    ee_loss = sum(losses.ee) / len(losses.ee)
    vec_loss = sum(losses.vec) / len(losses.vec)
    col_loss = sum(losses.col) / len(losses.col)
    lim_loss = sum(losses.lim) / len(losses.lim)
    ori_loss = sum(losses.ori) / len(losses.ori)
    fin_loss = sum(losses.fin) / len(losses.fin)
    reg_loss = sum(losses.reg) / len(losses.reg)
    # Log
    writer.add_scalars("training_loss", {"train": train_loss}, epoch + 1)
    writer.add_scalars("end_effector_loss", {"train": ee_loss}, epoch + 1)
    writer.add_scalars("vector_loss", {"train": vec_loss}, epoch + 1)
    writer.add_scalars("collision_loss", {"train": col_loss}, epoch + 1)
    writer.add_scalars("joint_limit_loss", {"train": lim_loss}, epoch + 1)
    writer.add_scalars("orientation_loss", {"train": ori_loss}, epoch + 1)
    writer.add_scalars("finger_loss", {"train": fin_loss}, epoch + 1)
    writer.add_scalars("regularization_loss", {"train": reg_loss}, epoch + 1)
    end_time = time.time()
    logger.info(
        "Epoch {:03d} | Training Time {:.2f} s | Avg Training {:.2f} | "
        "Avg EE {:.2f} | Avg Vec {:.2f} | Avg Col {:.2f} | Avg Lim {:.2f} | "
        "Avg Ori {:.2f} | Avg Fin {:.2f} | Avg Reg {:.2f}".format(
            epoch + 1,
            end_time - start_time,
            train_loss,
            ee_loss,
            vec_loss,
            col_loss,
            lim_loss,
            ori_loss,
            fin_loss,
            reg_loss,
        )
    )
    return train_loss
