import torch
from torch_geometric.data import Batch
from models.loss import calculate_all_loss, Loss
import time
from train import Model_Params
from loguru import logger


def log_data(
    writer,
    epoch: int,
    loss: Loss,
    start_time: float,
):
    if writer is not None:
        writer.add_scalars("testing_loss", {"test": loss.sum}, epoch + 1)
        writer.add_scalars("end_effector_loss", {"test": loss.ee}, epoch + 1)
        writer.add_scalars("vector_loss", {"test": loss.vec}, epoch + 1)
        writer.add_scalars("collision_loss", {"test": loss.col}, epoch + 1)
        writer.add_scalars("joint_limit_loss", {"test": loss.lim}, epoch + 1)
        writer.add_scalars("orientation_loss", {"test": loss.ori}, epoch + 1)
        writer.add_scalars("finger_loss", {"test": loss.fin}, epoch + 1)
        writer.add_scalars("regularization_loss", {"test": loss.reg}, epoch + 1)
    end_time = time.time()
    logger.info(
        "Testing Time {:.2f} s | Avg Testing {:.2f} | EE {:.2f} | "
        "Vec {:.2f} | Col {:.2f} | Lim {:.2f} | Ori {:.2f} | Fin {:.2f} | Reg {:.2f}".format(
            end_time - start_time,
            loss.sum,
            loss.ee,
            loss.vec,
            loss.col,
            loss.lim,
            loss.ori,
            loss.fin,
            loss.reg,
        )
    )


def record_actions(arm_data, hand_data, inferenced_data):
    inferenced_data["l_arm"].append(
        arm_data.arm_ang.squeeze().tolist()[: int(arm_data.arm_ang.size(0) / 2)]
    )
    inferenced_data["r_arm"].append(
        arm_data.arm_ang.squeeze().tolist()[int(arm_data.arm_ang.size(0) / 2) :]
    )
    inferenced_data["l_hand"].append(hand_data.l_hand_ang.squeeze().tolist()[1:])
    inferenced_data["r_hand"].append(hand_data.r_hand_ang.squeeze().tolist()[1:])
    return inferenced_data


def test_epoch(model_params: Model_Params, test_info_all=False, record_actions=False):
    epoch = model_params.epoch
    logger.info("Testing Epoch {}".format(epoch + 1))
    start_time = time.time()

    model_params.model.eval()
    losses = Loss()

    inferenced_data = {
        "l_arm": [],
        "r_arm": [],
        "l_hand": [],
        "r_hand": [],
        "loss": [],
    }
    with torch.no_grad():
        for _, data_list in enumerate(model_params.loader):
            for _, target in enumerate(model_params.target):
                # fetch target
                target_list = [target for _ in data_list]
                # forward
                z, arm_data, hand_data = model_params.model(
                    Batch.from_data_list(data_list).to(model_params.device),
                    Batch.from_data_list(target_list).to(model_params.device),
                )
                # calculate all loss
                _, losses = calculate_all_loss(
                    data_list,
                    target_list,
                    model_params.loss_criterion,
                    z,
                    arm_data,
                    hand_data,
                    losses,
                    model_params.loss_gain,
                )
                if record_actions:
                    inferenced_data = record_actions(
                        arm_data, hand_data, inferenced_data
                    )
            # log
            if test_info_all:
                logger.info(
                    "Sum {:.2f} | EE {:.2f} | Vec {:.2f} | Col {:.2f} | Lim {:.2f} | "
                    "Ori {:.2f} | Fin {:.2f} | Reg {:.2f}".format(
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
    test_loss = sum(losses.sum) / len(losses.sum)
    loss = Loss(
        sum=test_loss,
        ee=sum(losses.ee) / len(losses.ee),
        vec=sum(losses.vec) / len(losses.vec),
        col=sum(losses.col) / len(losses.col),
        lim=sum(losses.lim) / len(losses.lim),
        ori=sum(losses.ori) / len(losses.ori),
        fin=sum(losses.fin) / len(losses.fin),
        reg=sum(losses.reg) / len(losses.reg),
    )
    # Log
    log_data(model_params.writer, epoch, loss, start_time)

    if record_actions:
        return test_loss, inferenced_data
    else:
        return test_loss
