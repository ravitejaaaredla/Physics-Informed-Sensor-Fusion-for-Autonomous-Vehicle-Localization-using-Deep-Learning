import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from config import Config
from src.models import FusionModel
from src.eval.metrics import ATE, RPE

def load_raw_sequence_norm(seq_dir, config):
    root = Path(seq_dir)
    poses_file = root / "poses.txt"
    poses = np.loadtxt(poses_file).reshape(-1, 3, 4)
    num_frames = min(config.max_frames, len(poses))
    poses = poses[:num_frames]
    targets = []
    for i in range(num_frames):
        p = poses[i, :3, 3]
        Rmat = poses[i, :3, :3]
        yaw = np.arctan2(Rmat[1,0], Rmat[0,0])
        if i == 0:
            v = 0.0
        else:
            dt = 0.1
            dp = p - poses[i-1, :3, 3]
            v = np.linalg.norm(dp) / dt
        targets.append([p[0], p[1], v, yaw])
    targets = np.array(targets)
    pos_mean = np.mean(targets[:,:2], axis=0)
    pos_std = np.std(targets[:,:2], axis=0) + 1e-8
    targets_norm = targets.copy()
    targets_norm[:,:2] = (targets[:,:2] - pos_mean) / pos_std
    return targets_norm, targets, pos_mean, pos_std

def load_sensor_data(seq_dir, num_frames, config):
    root = Path(seq_dir)
    lidar_pts = []
    for i in range(num_frames):
        bin_file = root / "velodyne_points" / "data" / f"{i:010d}.bin"
        if bin_file.exists():
            pts = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)[:,:3]
            idx = np.random.choice(len(pts), min(1024, len(pts)), replace=False)
            pts = pts[idx]
            lidar_pts.append(pts.astype(np.float32))
        else:
            lidar_pts.append(np.random.randn(1024,3).astype(np.float32))
    cam_imgs = []
    for i in range(num_frames):
        img_file = root / "image_02" / "data" / f"{i:010d}.png"
        if img_file.exists():
            img = Image.open(img_file).resize((config.camera_img_w, config.camera_img_h))
            img = np.array(img, dtype=np.float32) / 255.0
            cam_imgs.append(img)
        else:
            cam_imgs.append(np.zeros((config.camera_img_h, config.camera_img_w, 3), dtype=np.float32))
    oxts_dir = root / "oxts" / "data"
    imu_data = []
    for i in range(num_frames):
        oxts_file = oxts_dir / f"{i:010d}.txt"
        if oxts_file.exists():
            oxts = np.loadtxt(oxts_file)
            imu = oxts[11:14].tolist() + oxts[17:20].tolist()
            imu_data.append(imu)
        else:
            imu_data.append([0,0,0,0,0,0])
    imu_data = np.array(imu_data, dtype=np.float32)
    return lidar_pts, cam_imgs, imu_data

def evaluate_model(model, imus, lidar_pts, cams, targets, device):
    imu_t = torch.tensor(imus, dtype=torch.float32).unsqueeze(1).to(device)
    lidar_t = torch.stack([torch.tensor(p, dtype=torch.float32) for p in lidar_pts]).to(device)
    cam_t = torch.stack([torch.tensor(c, dtype=torch.float32).permute(2,0,1) for c in cams]).to(device)
    model.eval()
    with torch.no_grad():
        pred_norm = model(imu_t, lidar_t, cam_t).cpu().numpy()
    return ATE(pred_norm[:,:2], targets[:,:2]), RPE(pred_norm[:,:2], targets[:,:2])

def add_imu_noise(imu, noise_std=0.05):
    return imu + np.random.normal(0, noise_std, imu.shape)

def occlude_lidar(points, keep_ratio=0.5):
    n = len(points)
    keep = np.random.choice(n, int(n*keep_ratio), replace=False)
    return points[keep]

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_dir = "data/Kitti_raw/2011_09_26-3/2011_09_26_drive_0009_sync"
    targets_norm, _, _, _ = load_raw_sequence_norm(seq_dir, config)
    lidar_pts, cam_imgs, imu_data = load_sensor_data(seq_dir, len(targets_norm), config)
    model = FusionModel(config).to(device)
    model.load_state_dict(torch.load("models/pinn_physics.pth", map_location=device))

    noise_levels = [0.05, 0.1, 0.2]
    noise_results = []
    for noise in tqdm(noise_levels, desc="Noise test"):
        imus_noisy = add_imu_noise(imu_data, noise)
        ate, rpe = evaluate_model(model, imus_noisy, lidar_pts, cam_imgs, targets_norm, device)
        noise_results.append({"noise_std": noise, "ATE": ate, "RPE": rpe})
    df_noise = pd.DataFrame(noise_results)

    keep_ratios = [1.0, 0.7, 0.5, 0.3]
    occ_results = []
    for ratio in tqdm(keep_ratios, desc="Occlusion test"):
        lidar_occ = [occlude_lidar(p, ratio) for p in lidar_pts]
        ate, rpe = evaluate_model(model, imu_data, lidar_occ, cam_imgs, targets_norm, device)
        occ_results.append({"keep_ratio": ratio, "ATE": ate, "RPE": rpe})
    df_occ = pd.DataFrame(occ_results)

    os.makedirs("results", exist_ok=True)
    df_noise.to_csv("results/noise_robustness_physics.csv", index=False)
    df_occ.to_csv("results/occlusion_robustness_physics.csv", index=False)
    print("Robustness tests saved to results/")

if __name__ == "__main__":
    main()