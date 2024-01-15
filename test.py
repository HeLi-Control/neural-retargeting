import torch
from torch_geometric.data import Batch
from models.loss import calculate_all_loss, Loss
import time
from train import Model_Params


def log_data(
    writer,
    logger,
    epoch: int,
    loss: Loss,
    start_time: float,
):
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
        "Epoch {:03d} | Testing Time {:.2f} s | Avg Testing {:.2f} | EE {:.2f} | "
        "Vec {:.2f} | Col {:.2f} | Lim {:.2f} | Ori {:.2f} | Fin {:.2f} | Reg {:.2f}".format(
            epoch + 1,
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


def test_epoch(model_params: Model_Params):
    epoch = model_params.epoch
    model_params.logger.info("Testing Epoch {}".format(epoch + 1).center(100, "-"))
    start_time = time.time()

    model_params.model.eval()
    losses = Loss()
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
    log_data(model_params.writer, model_params.logger, epoch, loss, start_time)

    return test_loss
