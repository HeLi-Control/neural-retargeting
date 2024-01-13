import math

import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataListLoader
from tensorboardX import SummaryWriter

from models import model
from models.loss import CollisionLoss, JointLimitLoss, RegLoss, Loss
import dataset
from dataset import *
from train import train_epoch
from test import test_epoch
from utils.config import cfg
from utils.util import create_folder

import os
import logging
import argparse
from datetime import datetime


def get_varying_learning_rate(
    given_learning_rate: float, varying_gain: float, epoch_now: float, epoch_sum: float
) -> float:
    return (
        given_learning_rate
        * (1 + varying_gain)
        / (1 + varying_gain * epoch_now / epoch_sum)
    )


# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Search for all yaml files in the path
    yaml_path = "configs/train/"
    print("Scanning processed files...")
    train_yaml = None
    params_yaml_files = []
    paths = os.walk(yaml_path)
    for path, dir_list, file_list in paths:
        for file in file_list:
            if file.endswith("train.yaml"):
                train_yaml = os.path.join(path, file).replace("\\", "/")
            elif file.endswith(".yaml"):
                params_yaml_files.append(os.path.join(path, file).replace("\\", "/"))
    # For all yaml files train a new net
    for yaml_file in params_yaml_files:
        print("<" * 80)
        print("Reading from yaml file:", yaml_file)

        # Set the best loss
        best_loss = float("Inf")

        # Argument parse
        parser = argparse.ArgumentParser(description="Command line arguments")
        parser.add_argument(
            "--cfg",
            default=yaml_file,
            type=str,
            help="Path to configuration file",
        )
        args = parser.parse_args()
        # Configurations parse
        cfg.merge_from_file(train_yaml)
        cfg.merge_from_file(args.cfg)
        cfg.freeze()
        # Create model
        model = getattr(model, cfg.MODEL.NAME)().to(device)
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=cfg.HYPER.LEARNING_RATE)
        # Load checkpoint
        if cfg.MODEL.CHECKPOINT is not None:
            model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))
            print("loaded checkpoint")

        # Create loss criterion
        loss_criterion = Loss(
            ee=nn.MSELoss() if cfg.LOSS.EE else None,
            vec=nn.MSELoss() if cfg.LOSS.VEC else None,
            col=CollisionLoss(cfg.LOSS.COL_THRESHOLD) if cfg.LOSS.COL else None,
            lim=JointLimitLoss() if cfg.LOSS.LIM else None,
            ori=nn.MSELoss() if cfg.LOSS.ORI else None,
            fin=nn.MSELoss() if cfg.LOSS.FIN else None,
            reg=RegLoss() if cfg.LOSS.REG else None
        )

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
            print("Times:", times + 1)
            print(">" * 80)

            # Refresh the best loss
            best_loss = best_loss * 1.5 if not math.isinf(best_loss) else best_loss

            # Network reset
            model.reset_parameters()

            # Create folder
            save = cfg.OTHERS.SAVE + "/{}".format(times + 1)
            log = cfg.OTHERS.LOG + "/{}".format(times + 1)
            summary = cfg.OTHERS.SUMMARY + "/{}".format(times + 1)
            create_folder(save)
            create_folder(log)
            create_folder(summary)
            # Create logger & tensorboard writer
            logging.basicConfig(
                level=logging.INFO,
                format="%(message)s",
                handlers=[
                    logging.FileHandler(
                        os.path.join(
                            log,
                            "{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now()),
                        )
                    ),
                    logging.StreamHandler(),
                ],
            )
            logger = logging.getLogger()
            writer = SummaryWriter(
                os.path.join(summary, "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now()))
            )

            # train
            for epoch in range(cfg.HYPER.EPOCHS):
                loss_gain = (
                    torch.tensor(cfg.LOSS.LOSS_GAIN)
                    if cfg.LOSS.LOSS_USING_GAIN
                    else None
                )
                # Set learning rate
                optimizer.lr = (
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
                train_loss = train_epoch(
                    model,
                    loss_criterion,
                    optimizer,
                    train_loader,
                    train_target,
                    epoch,
                    logger,
                    cfg.OTHERS.LOG_INTERVAL,
                    writer,
                    device,
                    loss_gain,
                )
                # Start testing
                test_loss = test_epoch(
                    model,
                    loss_criterion,
                    test_loader,
                    test_target,
                    epoch,
                    logger,
                    writer,
                    device,
                    loss_gain,
                )
                # Save model
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            save,
                            "best_model_epoch_{:04d}_loss_{:.2f}.pth".format(
                                epoch, best_loss
                            ),
                        ),
                    )
                    logger.info(
                        "Epoch {} Model Saved".format(epoch + 1).center(100, "-")
                    )
