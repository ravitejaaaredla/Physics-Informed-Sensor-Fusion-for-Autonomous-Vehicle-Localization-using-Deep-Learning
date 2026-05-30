import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from config import Config
from src.models import FusionModel
from src.physics_loss import PhysicsLoss
from src.eval.metrics import ATE, RPE
import os
from PIL import Image
from tqdm import tqdm

# -------------------------------------------------------------------
# Data loader for windows (real KITTI raw sequence)
# -------------------------------------------------------------------
def load_raw_sequence(config, seq_dir):
    root = Path(seq_dir)
    poses_file = root / "poses.txt"
    if not poses_file.exists():
        raise FileNotFoundError(f"Poses file not found: {poses_file}")

    poses = np.loadtxt(poses_file).reshape(-1, 3, 4)
    num_frames = min(config.max_frames, len(poses))
    poses = poses[:num_frames]

    # Targets: [x, y, v, yaw]
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

    # Normalise positions (zero‑mean, unit‑variance)
    pos_mean = np.mean(targets[:,:2], axis=0)
    pos_std = np.std(targets[:,:2], axis=0) + 1e-8
    targets[:,:2] = (targets[:,:2] - pos_mean) / pos_std

    # LiDAR
    velo_dir = root / "velodyne_points" / "data"
    lidar_pts = []
    for i in range(num_frames):
        bin_file = velo_dir / f"{i:010d}.bin"
        if bin_file.exists():
            pts = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)[:,:3]
            idx = np.random.choice(len(pts), min(1024, len(pts)), replace=False)
            pts = pts[idx]
            lidar_pts.append(pts.astype(np.float32))
        else:
            lidar_pts.append(np.random.randn(1024,3).astype(np.float32))

    # Camera
    img_dir = root / "image_02" / "data"
    cam_imgs = []
    for i in range(num_frames):
        img_file = img_dir / f"{i:010d}.png"
        if img_file.exists():
            img = Image.open(img_file).resize((config.camera_img_w, config.camera_img_h))
            img = np.array(img, dtype=np.float32) / 255.0
            cam_imgs.append(img)
        else:
            cam_imgs.append(np.zeros((config.camera_img_h, config.camera_img_w, 3), dtype=np.float32))

    # Real IMU from oxts
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

    return targets, lidar_pts, cam_imgs, imu_data, pos_mean, pos_std

class WindowDataset(Dataset):
    def __init__(self, targets, lidar_pts, cam_imgs, imu_data, window_size=10, stride=5):
        self.window_size = window_size
        self.stride = stride
        self.targets = targets
        self.lidar_pts = lidar_pts
        self.cam_imgs = cam_imgs
        self.imu_data = imu_data
        self.indices = list(range(0, len(targets) - window_size + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.window_size
        # targets: (win, 4)
        target_win = self.targets[start:end]
        # IMU: (win, 6)
        imu_win = self.imu_data[start:end]
        # LiDAR: list of (1024,3) for each frame
        lidar_win = self.lidar_pts[start:end]
        # Camera: list of (H,W,3) for each frame
        cam_win = self.cam_imgs[start:end]
        return (torch.tensor(imu_win, dtype=torch.float32),
                torch.stack([torch.tensor(p, dtype=torch.float32) for p in lidar_win]),
                torch.stack([torch.tensor(c, dtype=torch.float32).permute(2,0,1) for c in cam_win]),
                torch.tensor(target_win, dtype=torch.float32))

# -------------------------------------------------------------------
# Training loop with physics loss
# -------------------------------------------------------------------
def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_dir = "data/Kitti_raw/2011_09_26-3/2011_09_26_drive_0009_sync"
    print(f"Loading raw sequence from {seq_dir}...")
    targets, lidar_pts, cam_imgs, imu_data, pos_mean, pos_std = load_raw_sequence(config, seq_dir)

    dataset = WindowDataset(targets, lidar_pts, cam_imgs, imu_data,
                            window_size=10, stride=5)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = FusionModel(config).to(device)
    physics_loss_fn = PhysicsLoss(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    print("Training with real IMU, LiDAR, camera and physics‑informed loss (windowed)...")
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        for imu_win, lidar_win, cam_win, target_win in tqdm(loader, desc=f"Epoch {epoch+1}"):
            # imu_win: (batch, win, 6)
            # lidar_win: (batch, win, 1024, 3)
            # cam_win: (batch, win, 3, 128, 416)
            # target_win: (batch, win, 4)
            imu_win = imu_win.to(device)
            lidar_win = lidar_win.to(device)
            cam_win = cam_win.to(device)
            target_win = target_win.to(device)

            batch_size, win = imu_win.shape[:2]

            # Model expects (batch, seq_len, 6) for IMU, (batch, N, 3) for LiDAR, (batch, 3, H, W) for camera
            # But our model currently works only on single timestep. To use the existing model, we will iterate over timesteps.
            # We'll predict each timestep separately and stack the predictions.
            preds = []
            for t in range(win):
                imu_t = imu_win[:, t, :].unsqueeze(1)   # (batch, 1, 6)
                lidar_t = lidar_win[:, t, :, :]         # (batch, 1024, 3)
                cam_t = cam_win[:, t, :, :, :]          # (batch, 3, 128, 416)
                pred_t = model(imu_t, lidar_t, cam_t)   # (batch, 4)
                preds.append(pred_t.unsqueeze(1))
            pred_win = torch.cat(preds, dim=1)           # (batch, win, 4)

            # Compute data loss
            data_loss = torch.nn.MSELoss()(pred_win, target_win)

            # Compute physics loss (needs windowed predictions and IMU accelerations/gyro)
            # For simplicity, we use the same IMU data used as input (the first 6 columns are acc+gyro)
            imu_acc_body = imu_win[:, :, :3]    # (batch, win, 3)  accelerations
            imu_gyro_z = imu_win[:, :, 5]       # (batch, win)      yaw rate (gyro z)

            # PhysicsLoss expects (pred, target, imu_acc_body, imu_gyro_z)
            # Our PhysicsLoss.forward is designed for (B,T,4) etc. It will compute residuals over the window.
            phys_loss = physics_loss_fn(pred_win, target_win, imu_acc_body, imu_gyro_z)

            total = data_loss + phys_loss   # Adaptive weighting inside PhysicsLoss already handles lambdas
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            total_loss += total.item()

        print(f"Epoch {epoch+1}/{config.num_epochs} Avg Loss: {total_loss/len(loader):.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/pinn_physics.pth")
    print("Model saved as models/pinn_physics.pth")

    # Final evaluation on full sequence (non‑windowed, but normalised)
    # We'll evaluate frame‑by‑frame using the trained model.
    model.eval()
    imu_full = torch.tensor(imu_data, dtype=torch.float32).unsqueeze(1).to(device)
    lidar_full = torch.stack([torch.tensor(p, dtype=torch.float32) for p in lidar_pts]).to(device)
    cam_full = torch.stack([torch.tensor(c, dtype=torch.float32).permute(2,0,1) for c in cam_imgs]).to(device)
    target_full = torch.tensor(targets, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_full = model(imu_full, lidar_full, cam_full).cpu().numpy()
    # Denormalise positions
    pred_full[:,:2] = pred_full[:,:2] * pos_std + pos_mean
    targets_denorm = targets.copy()
    targets_denorm[:,:2] = targets_denorm[:,:2] * pos_std + pos_mean
    ate = ATE(pred_full[:,:2], targets_denorm[:,:2])
    rpe = RPE(pred_full[:,:2], targets_denorm[:,:2])
    print(f"Evaluation on full sequence (denormalised): ATE = {ate:.3f} m, RPE = {rpe:.3f} m")

if __name__ == "__main__":
    main()