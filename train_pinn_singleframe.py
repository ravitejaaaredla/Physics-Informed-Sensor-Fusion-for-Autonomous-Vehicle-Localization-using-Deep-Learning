import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from config import Config
from src.kitti_loader import load_kitti_sequence
from src.models import FusionModel
from src.physics_loss import PhysicsLoss
import os

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading synthetic IMU data (kitti_unified)...")
    bevs, imus, gps, cams, targets_norm, lidar_pts, pos_mean, pos_std = load_kitti_sequence(config)
    imu_t = torch.tensor(imus, dtype=torch.float32).unsqueeze(1)
    lidar_t = torch.stack([torch.tensor(p, dtype=torch.float32) for p in lidar_pts])
    cam_t = torch.stack([torch.tensor(c, dtype=torch.float32).permute(2,0,1) for c in cams])
    target_t = torch.tensor(targets_norm, dtype=torch.float32)
    dataset = TensorDataset(imu_t, lidar_t, cam_t, target_t)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = FusionModel(config).to(device)
    loss_fn = PhysicsLoss(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    print("Training PINN (single-frame, synthetic IMU)...")
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        for imu_b, lidar_b, cam_b, target_b in loader:
            imu_b, lidar_b, cam_b, target_b = [x.to(device) for x in [imu_b, lidar_b, cam_b, target_b]]
            pred = model(imu_b, lidar_b, cam_b)
            loss = loss_fn(pred, target_b, None, None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{config.num_epochs} Loss: {total_loss/len(loader):.6f}")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/pinn_singleframe.pth")
    print("Saved to models/pinn_singleframe.pth")

if __name__ == "__main__":
    main()