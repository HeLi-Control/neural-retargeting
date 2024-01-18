import torch
import torch.nn as nn
import torch_geometric.transforms as transforms
from torch_geometric.data import Batch
from torch_geometric.loader import DataListLoader
import argparse

from models.loss import calculate_all_loss, Loss

import dataset
from dataset import *
import models.model as model
from utils.config import cfg
from utils.util import create_folder

from tqdm import *
import h5py
import os

from loguru import logger


class Inference:
    def __init__(self, inference_calc_loss=False, device=torch.device("cpu")) -> None:
        self.device = device
        self.load_data()
        self.inferenced_data = {
            "l_arm": [],
            "r_arm": [],
            "l_hand": [],
            "r_hand": [],
            "loss": [],
        }
        self.inference_calc_loss = inference_calc_loss
        if self.inference_calc_loss:
            self.loss_criterion = Loss(
                ee=nn.MSELoss() if cfg.LOSS.EE else None,
                vec=nn.MSELoss() if cfg.LOSS.VEC else None,
                col=None,
                lim=None,
                ori=nn.MSELoss() if cfg.LOSS.ORI else None,
                fin=nn.MSELoss() if cfg.LOSS.FIN else None,
                reg=None,
            )

    def init_model(self, model_name: str):
        # Create model
        self.model = getattr(model, model_name)().to(self.device)

    def load_data(self):
        # Load data
        self.test_set = getattr(dataset, cfg.DATASET.TEST.SOURCE_NAME)(
            root=cfg.DATASET.TEST.SOURCE_PATH,
            pre_transform=transforms.Compose([Normalize()]),
        )
        self.test_loader = DataListLoader(
            self.test_set,
            batch_size=cfg.HYPER.BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        self.test_target_list = sorted(
            [
                target
                for target in getattr(dataset, cfg.DATASET.TEST.TARGET_NAME)(
                    root=cfg.DATASET.TEST.TARGET_PATH
                )
            ],
            key=lambda target: target.skeleton_type,
        )

    def load_trained_model(self, model_file: str):
        self.model.load_state_dict(torch.load(model_file))
        self.model.eval()

    def inference_frame(self, data_list):
        # forward
        _, arm_data, hand_data = self.model(
            Batch.from_data_list(data_list).to(self.device),
            Batch.from_data_list(self.test_target_list).to(self.device),
        )
        return arm_data, hand_data

    def save_inferenced_frame_data(
        self, arm_data, hand_data, data_list=None, print_loss=False
    ):
        # inferenced data
        self.inferenced_data["l_arm"].append(
            arm_data.arm_ang.squeeze().tolist()[: int(arm_data.arm_ang.size(0) / 2)]
        )
        self.inferenced_data["r_arm"].append(
            arm_data.arm_ang.squeeze().tolist()[int(arm_data.arm_ang.size(0) / 2) :]
        )
        self.inferenced_data["l_hand"].append(
            hand_data.l_hand_ang.squeeze().tolist()[1:]
        )
        self.inferenced_data["r_hand"].append(
            hand_data.r_hand_ang.squeeze().tolist()[1:]
        )

        new_loss = [0, 0, 0, 0]
        if self.inference_calc_loss:
            # loss calculation
            _, losses = calculate_all_loss(
                data_list,
                self.test_target_list,
                self.loss_criterion,
                None,
                arm_data,
                hand_data,
                None,
                torch.tensor(cfg.LOSS.LOSS_GAIN),
            )
            new_loss = [losses.ee[-1], losses.vec[-1], losses.ori[-1], losses.fin[-1]]
            if print_loss:
                logger.info(new_loss)
        self.inferenced_data["loss"].append(new_loss)

    def save_all_inferenced_data(self, save_path: str):
        h5_file = h5py.File(save_path, "w")
        if h5_file is not None:
            h5_file["l_arm"] = self.inferenced_data["l_arm"]
            h5_file["r_arm"] = self.inferenced_data["r_arm"]
            h5_file["l_hand"] = self.inferenced_data["l_hand"]
            h5_file["r_hand"] = self.inferenced_data["r_hand"]
            h5_file["loss"] = self.inferenced_data["loss"]
        h5_file.close()
        # Clear inferenced data
        self.inferenced_data = {
            "l_arm": [],
            "r_arm": [],
            "l_hand": [],
            "r_hand": [],
            "loss": [],
        }


def search_checkpoints(checkpoints_path: str):
    # Search checkpoints
    paths = os.walk(checkpoints_path)
    inference_model_files = []
    for path, _, file_list in paths:
        for file in file_list:
            if file.endswith(".pth"):
                inference_model_files.append(
                    os.path.join(path, file).replace("\\", "/")
                )
    return inference_model_files


def save_demonstrate_actions(save_file_name: str, test_loader):
    human_demonstrate_data = {
        "l_arm": [],
        "r_arm": [],
        "l_hand": [],
        "r_hand": [],
        "ee_ori": [],
    }
    for _, data_list in tqdm(
        enumerate(test_loader),
        total=len(test_loader),
    ):
        human_demonstrate_data["l_arm"].append(data_list[0].pos[:3].tolist())
        human_demonstrate_data["r_arm"].append(data_list[0].pos[3:].tolist())
        human_demonstrate_data["l_hand"].append(data_list[0].l_hand_pos.tolist())
        human_demonstrate_data["r_hand"].append(data_list[0].r_hand_pos.tolist())
        human_demonstrate_data["ee_ori"].append(
            [data_list[0].q[2].tolist(), data_list[0].q[5].tolist()]
        )
    h5_file = h5py.File(save_file_name, "w")
    h5_file["l_arm"] = human_demonstrate_data["l_arm"]
    h5_file["r_arm"] = human_demonstrate_data["r_arm"]
    h5_file["l_hand"] = human_demonstrate_data["l_hand"]
    h5_file["r_hand"] = human_demonstrate_data["r_hand"]
    h5_file["ee_ori"] = human_demonstrate_data["ee_ori"]
    h5_file.close()


if __name__ == "__main__":
    # Argument parse
    parser = argparse.ArgumentParser(description="Inference with trained model")
    parser.add_argument(
        "--cfg",
        default="configs/inference/inference.yaml",
        type=str,
        help="Path to configuration file",
    )
    args = parser.parse_args()
    # Configurations parse
    cfg.merge_from_file("configs/global.yaml")
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    # Inference preparation
    inference = Inference(
        inference_calc_loss=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    create_folder(cfg.INFERENCE.H5.PATH)
    # Save source data
    if cfg.INFERENCE.RUN.HUMAN_DEMONSTRATE:
        logger.debug("Scanning source data...")
        file_name = (
            cfg.INFERENCE.H5.PATH + "humanDemonstrate_" + cfg.DATASET.TYPE + ".h5"
        )
        save_demonstrate_actions(file_name, inference.test_loader)
        logger.debug("Saved demonstrate data to file:" + file_name)
    # Inference
    if cfg.INFERENCE.RUN.INFERENCE:
        # Start simulation
        if cfg.INFERENCE.PYBULLET.BOOL:
            from simulation import YuMi_Simulation
        yumi_sim = (
            YuMi_Simulation(
                disp_gui=True,
                debug_message_cfg={"draw_debug": False, "print_debug": False},
            )
            if cfg.INFERENCE.PYBULLET.BOOL
            else None
        )
        logger.debug("Inferencing.")
        inference.init_model(cfg.MODEL.NAME)
        inference_model_files = search_checkpoints(cfg.MODEL.CHECKPOINT)
        for model_file in inference_model_files:
            # load model
            inference.load_trained_model(model_file)
            logger.info("Loaded model:" + model_file)
            # Inference
            with torch.no_grad():
                for _, data_list in tqdm(
                    enumerate(inference.test_loader),
                    total=len(inference.test_loader),
                    leave=True,
                ):
                    # Inference a frame
                    arm_data, hand_data = inference.inference_frame(data_list)
                    # Save data
                    inference.save_inferenced_frame_data(
                        arm_data, hand_data, data_list, print_loss=True
                    )
                    # Step the simulation
                    if cfg.INFERENCE.PYBULLET.BOOL:
                        data_frame = (
                            inference.inferenced_data["l_arm"][-1]
                            + inference.inferenced_data["l_hand"][-1]
                            + inference.inferenced_data["r_arm"][-1]
                            + inference.inferenced_data["r_hand"][-1]
                        )
                        yumi_sim.step_simulation(given_joints_angles=data_frame)
            # Save the model inferenced data
            if cfg.INFERENCE.H5.BOOL:
                inference.save_all_inferenced_data(
                    cfg.INFERENCE.H5.PATH
                    + os.path.basename(model_file)[:-4]
                    + "_"
                    + cfg.DATASET.TYPE
                    + ".h5"
                )
        if cfg.INFERENCE.PYBULLET.BOOL:
            # Stop the simulation
            yumi_sim.close_simulation()
            del yumi_sim
