# train_pure_delta.py

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from src.models import FusionModel
from src.raw_kitti_utils import load_raw_sequence, DeltaWindowDataset


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    config = Config()
    seed_everything(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    states, lidar_pts, cam_imgs, imu_data = load_raw_sequence(config.seq_dir, config)

    dataset = DeltaWindowDataset(
        states=states,
        lidar_pts=lidar_pts,
        cam_imgs=cam_imgs,
        imu_data=imu_data,
        config=config,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    print(f"Training samples: {len(dataset)}")

    model = FusionModel(config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    print("Training PURE-DL delta model...")
    print("Output target = [dx_body, dy_body, dv, dyaw]")

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        for imu_win, lidar, cam, prev_state, delta_target in tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
        ):
            imu_win = imu_win.to(device)
            lidar = lidar.to(device)
            cam = cam.to(device)
            delta_target = delta_target.to(device)

            pred_delta = model(imu_win, lidar, cam)

            loss = F.smooth_l1_loss(pred_delta, delta_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch + 1}/{config.num_epochs} | Loss: {avg_loss:.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/pure_delta.pth")
    print("Saved model to models/pure_delta.pth")


if __name__ == "__main__":
    main()