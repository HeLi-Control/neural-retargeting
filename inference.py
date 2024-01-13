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

# Argument parse
parser = argparse.ArgumentParser(description="Inference with trained model")
parser.add_argument(
    "--cfg",
    default="configs/inference/yumi.yaml",
    type=str,
    help="Path to configuration file",
)
args = parser.parse_args()

# Configurations parse
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

    # # New data file
    # h5_file = h5py.File(cfg.INFERENCE.H5.PATH, "w") if cfg.INFERENCE.H5.BOOL else None
    # if h5_file is not None:
    #     hand_set = h5_file.create_dataset("hand", (100,), dtype="float32")
    #     arm_set = h5_file.create_dataset("arm", (100,), dtype="float32")

    # Load checkpoint
    if cfg.MODEL.CHECKPOINT is not None:
        model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))
        print("Load model")

    # Inference
    print("Model evaluating...")
    model.eval()
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
            # if h5_file is not None:

    # # save data
    # print("Data saving...")
