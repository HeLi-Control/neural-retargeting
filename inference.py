import torch
import torch_geometric.transforms as transforms
from torch_geometric.data import Batch
from torch_geometric.loader import DataListLoader
import argparse

import dataset
from dataset import Normalize
import models.model as model
from utils.config import cfg
from utils.util import create_folder

from tqdm import *
import h5py

import os

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

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Load data
    pre_transform = transforms.Compose([Normalize()])
    # inference all
    print("Inference all")
    test_set = getattr(dataset, cfg.DATASET.TEST.SOURCE_NAME)(
        root=cfg.DATASET.TEST.SOURCE_PATH, pre_transform=pre_transform
    )
    test_loader = DataListLoader(
        test_set,
        batch_size=cfg.HYPER.BATCH_SIZE,
        shuffle=False,
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

    # Create model
    model = getattr(model, cfg.MODEL.NAME)().to(device)

    # Load checkpoints
    if cfg.MODEL.CHECKPOINT is not None:
        checkpoints_path = cfg.MODEL.CHECKPOINT
        paths = os.walk(checkpoints_path)
        inference_model_files = []
        for path, _, file_list in paths:
            for file in file_list:
                if file.endswith(".pth"):
                    inference_model_files.append(
                        os.path.join(path, file).replace("\\", "/")
                    )
        if len(inference_model_files) == 0:
            print("No models to inference.")
            exit(1)
    else:
        print("Checkpoint path unspecified.")
        exit(1)

    inferenced_data = (
        {"l_arm": [], "r_arm": [], "l_hand": [], "r_hand": []}
        if cfg.INFERENCE.H5.BOOL
        else None
    )
    create_folder(cfg.INFERENCE.H5.PATH)

    # Inference
    for model_file in inference_model_files:
        print("Loading model:", model_file)
        model.load_state_dict(torch.load(model_file))
        print("Model evaluating...")
        model.eval()
        print("Inferencing...")
        for batch_idx, data_list in tqdm(
            enumerate(test_loader), total=len(test_loader), leave=False
        ):
            for target_idx, target in enumerate(test_target):
                target_list = [target for _ in data_list]
                # forward
                _, arm_data, hand_data = model(
                    Batch.from_data_list(data_list).to(device),
                    Batch.from_data_list(target_list).to(device),
                )
                if cfg.INFERENCE.H5.BOOL:
                    inferenced_data["l_arm"].append(
                        arm_data.arm_ang.squeeze().tolist()[
                            : int(arm_data.arm_ang.size(0) / 2)
                        ]
                    )
                    inferenced_data["r_arm"].append(
                        arm_data.arm_ang.squeeze().tolist()[
                            int(arm_data.arm_ang.size(0) / 2) :
                        ]
                    )
                    inferenced_data["l_hand"].append(
                        hand_data.l_hand_ang.squeeze().tolist()[1:]
                    )
                    inferenced_data["r_hand"].append(
                        hand_data.r_hand_ang.squeeze().tolist()[1:]
                    )

        if cfg.INFERENCE.H5.BOOL:
            # save data
            print("Data saving...")
            # New data file
            save_file_name = os.path.basename(model_file)[:-4] + ".h5"
            h5_file = h5py.File(cfg.INFERENCE.H5.PATH + save_file_name, "w")
            if h5_file is not None:
                h5_file["l_arm"] = inferenced_data["l_arm"]
                h5_file["r_arm"] = inferenced_data["r_arm"]
                h5_file["l_hand"] = inferenced_data["l_hand"]
                h5_file["r_hand"] = inferenced_data["r_hand"]
            h5_file.close()
