import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataListLoader
from tensorboardX import SummaryWriter

from models import model
from models.loss import CollisionLoss, JointLimitLoss, RegLoss, Loss
import dataset
from dataset import *
from train import train_epoch, Model_Params
from test import test_epoch
from utils.config import cfg
from utils.util import create_folder

import os
import argparse
from datetime import datetime

from loguru import logger


def get_varying_learning_rate(
    given_learning_rate: float, varying_gain: float, epoch_now: float, epoch_sum: float
) -> float:
    return (
        given_learning_rate
        * (1 + varying_gain)
        / (1 + varying_gain * epoch_now / epoch_sum)
    )


def save_model(
    train_model: model,
    save_path: str,
    epoch_now: int,
    best_loss_now: float,
):
    torch.save(
        train_model.state_dict(),
        os.path.join(
            save_path,
            "best_model_epoch_{:03d}_loss_{:.2f}.pth".format(epoch_now, best_loss_now),
        ),
    )
    logger.info("Epoch {} Model Saved".format(epoch_now + 1))


def get_model_params(train_loader, train_target, device: torch.device, cfg):
    def mse(x: torch.Tensor, y: torch.Tensor):
        return torch.mean(torch.tensor([torch.norm(e).pow(2) for e in (x - y)]))

    from models import model

    train_model = getattr(model, cfg.MODEL.NAME)().to(device)
    model_params = Model_Params(
        model=train_model,
        optimizer=optim.Adam(train_model.parameters(), lr=cfg.HYPER.LEARNING_RATE),
        loss_criterion=Loss(
            ee=mse if cfg.LOSS.EE else None,
            vec=mse if cfg.LOSS.VEC else None,
            col=CollisionLoss(cfg.LOSS.COL_THRESHOLD) if cfg.LOSS.COL else None,
            lim=JointLimitLoss() if cfg.LOSS.LIM else None,
            ori=mse if cfg.LOSS.ORI else None,
            fin=mse if cfg.LOSS.FIN else None,
            reg=RegLoss() if cfg.LOSS.REG else None,
        ),
        loader=train_loader,
        target=train_target,
        epoch=0,
        interval=cfg.OTHERS.LOG_INTERVAL,
        writer=None,
        device=device,
        loss_gain=torch.tensor(cfg.LOSS.LOSS_GAIN)
        if cfg.LOSS.LOSS_USING_GAIN
        else None,
    )
    return model_params


if __name__ == "__main__":
    # Argument parse
    parser = argparse.ArgumentParser(description="Command line arguments")
    parser.add_argument(
        "--cfg",
        type=str,
        help="Path to configuration file",
    )
    args = parser.parse_args()
    cfg.merge_from_file("configs/global.yaml")
    cfg.merge_from_file("configs/train/train.yaml")
    # Search for all yaml files in the path
    yaml_path = "configs/train/"
    logger.debug("Scanning processed files...")
    params_yaml_files = []
    paths = os.walk(yaml_path)
    for path, dir_list, file_list in paths:
        for file in file_list:
            if cfg.TRAIN.READ_SPECIFIC_YAML:
                if cfg.TRAIN.SPECIFIC_YAML_NAME in file:
                    params_yaml_files.append(
                        os.path.join(path, file).replace("\\", "/")
                    )
            else:
                if file.endswith(".yaml") and not file.endswith("train.yaml"):
                    params_yaml_files.append(
                        os.path.join(path, file).replace("\\", "/")
                    )
    # For all yaml files train a new net
    for yaml_file in params_yaml_files:
        logger.info("Read from yaml file:" + yaml_file)

        cfg.merge_from_file(yaml_file)
        cfg.freeze()

        # Load data
        pre_transform = transforms.Compose([Normalize()])
        train_set = getattr(dataset, cfg.DATASET.TRAIN.SOURCE_NAME)(
            root=cfg.DATASET.TRAIN.SOURCE_PATH, pre_transform=pre_transform
        )
        train_loader = DataListLoader(
            train_set,
            batch_size=cfg.HYPER.BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        train_target = sorted(
            [
                target
                for target in getattr(dataset, cfg.DATASET.TRAIN.TARGET_NAME)(
                    root=cfg.DATASET.TRAIN.TARGET_PATH
                )
            ],
            key=lambda target: target.skeleton_type,
        )
        test_set = getattr(dataset, cfg.DATASET.TEST.SOURCE_NAME)(
            root=cfg.DATASET.TEST.SOURCE_PATH, pre_transform=pre_transform
        )
        test_loader = DataListLoader(
            test_set,
            batch_size=cfg.HYPER.BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        test_target = sorted(
            [
                target
                for target in getattr(dataset, cfg.DATASET.TEST.TARGET_NAME)(
                    root=cfg.DATASET.TEST.TARGET_PATH
                )
            ],
            key=lambda target: target.skeleton_type,
        )

        for times in range(cfg.HYPER.TRAIN_TIMES):
            logger.info("Times:{}".format(times + 1))
            # set model params
            model_params = get_model_params(
                train_loader,
                train_target,
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                cfg,
            )
            if cfg.MODEL.LOAD_CHECKPOINT:
                if cfg.MODEL.CHECKPOINT_RANDOM:
                    checkpoints = []
                    paths = os.walk(cfg.MODEL.CHECKPOINT)
                    for path, dir_list, file_list in paths:
                        for file in file_list:
                            if file.endswith(".pth"):
                                checkpoints.append(
                                    os.path.join(path, file).replace("\\", "/")
                                )
                    import random

                    checkpoint = random.choice(checkpoints)
                    model_params.model.load_state_dict(torch.load(checkpoint))
                    logger.info("Loaded trained model:" + checkpoint)
                else:
                    model_params.model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))
                    logger.info("Loaded trained model:" + cfg.MODEL.CHECKPOINT)
            # Create folder
            save = cfg.OTHERS.SAVE + "/{}".format(times + 1)
            summary = cfg.OTHERS.SUMMARY + "/{}".format(times + 1)
            log = cfg.OTHERS.LOG + "/{}".format(times + 1)
            create_folder(save)
            create_folder(summary)
            create_folder(log)
            logger.add(sink=log + "/train_{time}.log", rotation="5MB")
            model_params.writer = SummaryWriter(
                os.path.join(summary, "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now()))
            )
            # Set the best loss
            best_loss = float("Inf")
            # train
            for epoch in range(cfg.HYPER.EPOCHS):
                model_params.epoch = epoch
                logger.info("Epoch " + str(epoch + 1))
                # Set learning rate
                model_params.optimizer.lr = (
                    get_varying_learning_rate(
                        cfg.HYPER.LEARNING_RATE,
                        cfg.HYPER.VARIABLE_LEARNING_RATE_GAIN,
                        epoch,
                        cfg.HYPER.EPOCHS,
                    )
                    if cfg.HYPER.VARIABLE_LEARNING_RATE
                    else cfg.HYPER.LEARNING_RATE
                )
                # Start training
                model_params.loader = train_loader
                model_params.target = train_target
                train_loss = train_epoch(model_params)
                # Start testing
                model_params.loader = test_loader
                model_params.target = test_target
                test_loss = test_epoch(model_params)
                # Save model
                if test_loss < best_loss:
                    best_loss = test_loss
                    save_model(
                        model_params.model,
                        save,
                        epoch,
                        best_loss,
                    )
            del model_params
            best_loss = float("Inf")
