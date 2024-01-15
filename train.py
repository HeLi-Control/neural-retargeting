import torch
from torch_geometric.data import Batch
from models.loss import calculate_all_loss, Loss
import logging
from models.model import YumiNet
from tensorboardX import SummaryWriter
import time


class Model_Params:
    def __init__(
        self,
        model: YumiNet,
        loss_criterion: Loss,
        optimizer: torch.optim.Adam,
        loader,
        target,
        epoch: int,
        logger: logging.Logger,
        interval: int,
        writer,
        loss_gain: torch.Tensor,
        device: torch.device,
        z_all=None,
    ):
        self.model = model
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.loader = loader
        self.target = target
        self.epoch = epoch
        self.logger = logger
        self.interval = interval
        self.writer = writer
        self.device = device
        self.loss_gain = loss_gain
        self.z_all = z_all


def train_epoch(model_params: Model_Params):
    epoch = model_params.epoch
    model_params.logger.info("Training Epoch {}".format(epoch + 1).center(100, "-"))
    start_time = time.time()

    model_params.model.train()
    losses = Loss()
    for batch_idx, data_list in enumerate(model_params.loader):
        for _, target in enumerate(model_params.target):
            # zero gradient
            model_params.optimizer.zero_grad()
            # fetch target
            target_list = [target for _ in data_list]
            # forward
            if model_params.z_all is not None:
                z = model_params.z_all[batch_idx]
                _, arm_data, hand_data = model_params.model.decode(
                    z, Batch.from_data_list(target_list).to(model_params.device)
                )
            else:
                z, arm_data, hand_data = model_params.model(
                    Batch.from_data_list(data_list).to(model_params.device),
                    Batch.from_data_list(target_list).to(model_params.device),
                )
            # calculate all loss
            loss, losses = calculate_all_loss(
                data_list,
                target_list,
                model_params.loss_criterion,
                z,
                arm_data,
                hand_data,
                losses,
                model_params.loss_gain,
            )
            # backward
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model_params.model.parameters(), 10)
            # optimize
            model_params.optimizer.step()
        # log
        if (batch_idx + 1) % model_params.interval == 0:
            model_params.logger.info(
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
    model_params.writer.add_scalars("training_loss", {"train": train_loss}, epoch + 1)
    model_params.writer.add_scalars("end_effector_loss", {"train": ee_loss}, epoch + 1)
    model_params.writer.add_scalars("vector_loss", {"train": vec_loss}, epoch + 1)
    model_params.writer.add_scalars("collision_loss", {"train": col_loss}, epoch + 1)
    model_params.writer.add_scalars("joint_limit_loss", {"train": lim_loss}, epoch + 1)
    model_params.writer.add_scalars("orientation_loss", {"train": ori_loss}, epoch + 1)
    model_params.writer.add_scalars("finger_loss", {"train": fin_loss}, epoch + 1)
    model_params.writer.add_scalars(
        "regularization_loss", {"train": reg_loss}, epoch + 1
    )
    end_time = time.time()
    model_params.logger.info(
        "Epoch {:03d} | Training Time {:.2f} s | Avg Training {:.2f} | EE {:.2f} | Vec {:.2f} "
        "| Col {:.2f} | Lim {:.2f} | Ori {:.2f} | Fin {:.2f} | Reg {:.2f}".format(
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
