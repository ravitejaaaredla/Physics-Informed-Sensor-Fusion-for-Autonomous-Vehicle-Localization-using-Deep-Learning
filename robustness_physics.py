# robustness_physics.py

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from config import Config
from src.models import FusionModel


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def ate(pred_xy, gt_xy):
    n = min(len(pred_xy), len(gt_xy))
    pred_xy = pred_xy[:n]
    gt_xy = gt_xy[:n]
    return float(np.sqrt(np.mean(np.sum((pred_xy - gt_xy) ** 2, axis=1))))


def rpe(pred_xy, gt_xy):
    n = min(len(pred_xy), len(gt_xy))
    pred_xy = pred_xy[:n]
    gt_xy = gt_xy[:n]

    if n < 2:
        return float("nan")

    pred_rel = pred_xy[1:] - pred_xy[:-1]
    gt_rel = gt_xy[1:] - gt_xy[:-1]
    return float(np.sqrt(np.mean(np.sum((pred_rel - gt_rel) ** 2, axis=1))))


def load_raw_sequence(seq_dir, config):
    root = Path(seq_dir)

    poses_file = root / "poses.txt"
    if not poses_file.exists():
        raise FileNotFoundError(f"poses.txt not found: {poses_file}")

    poses = np.loadtxt(poses_file).reshape(-1, 3, 4)
    num_frames = min(config.max_frames, len(poses))
    poses = poses[:num_frames]

    states = []

    for i in range(num_frames):
        p = poses[i, :3, 3]
        R = poses[i, :3, :3]

        x = p[0]
        y = p[1]
        yaw = np.arctan2(R[1, 0], R[0, 0])

        if i == 0:
            v = 0.0
        else:
            dp = poses[i, :3, 3] - poses[i - 1, :3, 3]
            v = np.linalg.norm(dp) / config.dt

        states.append([x, y, v, yaw])

    states = np.asarray(states, dtype=np.float32)

    lidar_pts = []
    velo_dir = root / "velodyne_points" / "data"

    for i in range(num_frames):
        bin_file = velo_dir / f"{i:010d}.bin"
        if not bin_file.exists():
            raise FileNotFoundError(f"Missing LiDAR file: {bin_file}")

        pts = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)[:, :3]

        if len(pts) >= config.lidar_num_points:
            idx = np.random.choice(len(pts), config.lidar_num_points, replace=False)
        else:
            idx = np.random.choice(len(pts), config.lidar_num_points, replace=True)

        lidar_pts.append(pts[idx].astype(np.float32))

    cam_imgs = []
    img_dir = root / "image_02" / "data"

    for i in range(num_frames):
        img_file = img_dir / f"{i:010d}.png"
        if not img_file.exists():
            raise FileNotFoundError(f"Missing camera file: {img_file}")

        img = Image.open(img_file).convert("RGB")
        img = img.resize((config.camera_img_w, config.camera_img_h))
        img = np.asarray(img, dtype=np.float32) / 255.0
        cam_imgs.append(img)

    imu_data = []
    oxts_dir = root / "oxts" / "data"

    for i in range(num_frames):
        oxts_file = oxts_dir / f"{i:010d}.txt"
        if not oxts_file.exists():
            raise FileNotFoundError(f"Missing OXTS file: {oxts_file}")

        oxts = np.loadtxt(oxts_file)
        imu = oxts[11:14].tolist() + oxts[17:20].tolist()
        imu_data.append(imu)

    imu_data = np.asarray(imu_data, dtype=np.float32)

    return states, lidar_pts, cam_imgs, imu_data


def integrate_delta_model(model, states, lidar_pts, cam_imgs, imu_data, config, device):
    model.eval()

    pred_states = []

    x, y, v, yaw = states[0]
    pred_states.append([x, y, v, yaw])

    for end in range(config.window_size - 1, len(states) - 1):
        start = end - config.window_size + 1

        imu_win = imu_data[start : end + 1]
        lidar = lidar_pts[end]
        cam = cam_imgs[end]

        imu_t = torch.tensor(imu_win, dtype=torch.float32).unsqueeze(0).to(device)
        lidar_t = torch.tensor(lidar, dtype=torch.float32).unsqueeze(0).to(device)
        cam_t = (
            torch.tensor(cam, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            delta = model(imu_t, lidar_t, cam_t).cpu().numpy()[0]

        dx_b, dy_b, dv, dyaw = delta

        c = np.cos(yaw)
        s = np.sin(yaw)

        dx_w = c * dx_b - s * dy_b
        dy_w = s * dx_b + c * dy_b

        x = x + dx_w
        y = y + dy_w
        v = max(0.0, v + dv)
        yaw = wrap_angle(yaw + dyaw)

        pred_states.append([x, y, v, yaw])

    return np.asarray(pred_states, dtype=np.float32)


def add_imu_noise(imu_data, noise_std):
    return imu_data + np.random.normal(0.0, noise_std, size=imu_data.shape).astype(np.float32)


def occlude_lidar(lidar_pts, keep_ratio, config):
    occluded = []

    for pts in lidar_pts:
        n = len(pts)
        keep_n = max(8, int(n * keep_ratio))
        idx = np.random.choice(n, keep_n, replace=False)
        kept = pts[idx]

        if len(kept) >= config.lidar_num_points:
            idx2 = np.random.choice(len(kept), config.lidar_num_points, replace=False)
        else:
            idx2 = np.random.choice(len(kept), config.lidar_num_points, replace=True)

        occluded.append(kept[idx2].astype(np.float32))

    return occluded


def evaluate(model, states, lidar_pts, cam_imgs, imu_data, config, device):
    pred_states = integrate_delta_model(
        model,
        states,
        lidar_pts,
        cam_imgs,
        imu_data,
        config,
        device,
    )

    pred_xy = pred_states[:, :2]
    gt_xy = states[: len(pred_xy), :2]

    return ate(pred_xy, gt_xy), rpe(pred_xy, gt_xy)


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    states, lidar_pts, cam_imgs, imu_data = load_raw_sequence(config.seq_dir, config)

    model_path = "models/pinn_delta_physics.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Run python train_physics_loss.py first."
        )

    model = FusionModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------
    # IMU noise robustness
    # ------------------------------------------------------------
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    noise_rows = []

    for noise in tqdm(noise_levels, desc="IMU noise robustness"):
        noisy_imu = add_imu_noise(imu_data, noise)
        test_ate, test_rpe = evaluate(
            model,
            states,
            lidar_pts,
            cam_imgs,
            noisy_imu,
            config,
            device,
        )

        noise_rows.append(
            {
                "imu_noise_std": noise,
                "ATE_m": test_ate,
                "RPE_m": test_rpe,
            }
        )

    df_noise = pd.DataFrame(noise_rows)
    df_noise.to_csv("results/noise_robustness_physics.csv", index=False)

    # ------------------------------------------------------------
    # LiDAR occlusion robustness
    # ------------------------------------------------------------
    keep_ratios = [1.0, 0.7, 0.5, 0.3]
    occ_rows = []

    for keep_ratio in tqdm(keep_ratios, desc="LiDAR occlusion robustness"):
        lidar_occ = occlude_lidar(deepcopy(lidar_pts), keep_ratio, config)

        test_ate, test_rpe = evaluate(
            model,
            states,
            lidar_occ,
            cam_imgs,
            imu_data,
            config,
            device,
        )

        occ_rows.append(
            {
                "lidar_keep_ratio": keep_ratio,
                "ATE_m": test_ate,
                "RPE_m": test_rpe,
            }
        )

    df_occ = pd.DataFrame(occ_rows)
    df_occ.to_csv("results/occlusion_robustness_physics.csv", index=False)

    print("Saved:")
    print("  results/noise_robustness_physics.csv")
    print("  results/occlusion_robustness_physics.csv")


if __name__ == "__main__":
    main()