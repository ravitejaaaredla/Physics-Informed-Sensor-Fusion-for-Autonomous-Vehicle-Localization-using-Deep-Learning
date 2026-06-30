# train_physics_loss.py

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


def wrap_angle_torch(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class PhysicsDeltaLoss(torch.nn.Module):
    """
    Model output:
        pred_delta = [dx_body, dy_body, dv, dyaw]

    Physics constraints:
        1. dv should agree with forward acceleration.
        2. dyaw should agree with gyro z.
        3. displacement should agree with v*dt + 0.5*a*dt^2.
        4. lateral displacement should be small.
    """

    def __init__(self, config):
        super().__init__()

        self.dt = config.dt
        self.lambda_data = config.lambda_data
        self.lambda_physics = config.lambda_physics
        self.lambda_smooth = config.lambda_smooth

    def forward(self, pred_delta, target_delta, imu_win, prev_state):
        data_loss = F.smooth_l1_loss(pred_delta, target_delta)

        acc_x = imu_win[:, -1, 0]
        gyro_z = imu_win[:, -1, 5]
        v_prev = prev_state[:, 2]

        dx_b = pred_delta[:, 0]
        dy_b = pred_delta[:, 1]
        dv = pred_delta[:, 2]
        dyaw = pred_delta[:, 3]

        expected_dv = acc_x * self.dt
        expected_dyaw = gyro_z * self.dt

        expected_distance = torch.clamp(
            v_prev * self.dt + 0.5 * acc_x * (self.dt ** 2),
            min=0.0,
        )

        pred_distance = torch.sqrt(dx_b ** 2 + dy_b ** 2 + 1e-8)

        loss_dv = F.smooth_l1_loss(dv, expected_dv)
        loss_dyaw = F.smooth_l1_loss(
            wrap_angle_torch(dyaw),
            wrap_angle_torch(expected_dyaw),
        )
        loss_distance = F.smooth_l1_loss(pred_distance, expected_distance)

        # Non-holonomic vehicle constraint
        loss_lateral = torch.mean(torch.abs(dy_b))

        physics_loss = (
            loss_dv
            + loss_dyaw
            + loss_distance
            + 0.1 * loss_lateral
        )

        smooth_loss = torch.mean(pred_delta ** 2)

        total = (
            self.lambda_data * data_loss
            + self.lambda_physics * physics_loss
            + self.lambda_smooth * smooth_loss
        )

        return total, data_loss.detach(), physics_loss.detach(), smooth_loss.detach()


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

    loss_fn = PhysicsDeltaLoss(config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    print("Training PINN delta model...")
    print("Output target = [dx_body, dy_body, dv, dyaw]")

    for epoch in range(config.num_epochs):
        model.train()

        total_loss = 0.0
        total_data = 0.0
        total_phys = 0.0
        total_smooth = 0.0

        for imu_win, lidar, cam, prev_state, delta_target in tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
        ):
            imu_win = imu_win.to(device)
            lidar = lidar.to(device)
            cam = cam.to(device)
            prev_state = prev_state.to(device)
            delta_target = delta_target.to(device)

            pred_delta = model(imu_win, lidar, cam)

            loss, data_loss, phys_loss, smooth_loss = loss_fn(
                pred_delta,
                delta_target,
                imu_win,
                prev_state,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_data += data_loss.item()
            total_phys += phys_loss.item()
            total_smooth += smooth_loss.item()

        n = max(1, len(loader))

        print(
            f"Epoch {epoch + 1}/{config.num_epochs} | "
            f"Total: {total_loss / n:.6f} | "
            f"Data: {total_data / n:.6f} | "
            f"Physics: {total_phys / n:.6f} | "
            f"Smooth: {total_smooth / n:.6f}"
        )

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/pinn_delta_physics.pth")
    print("Saved model to models/pinn_delta_physics.pth")


if __name__ == "__main__":
    main()