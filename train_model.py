import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from config import Config
from src.kitti_loader import load_kitti_sequence
from src.models import FusionModel
from src.eval.metrics import ATE, RPE
import synthetic_data
import os

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading data...")
    try:
        bevs, imus, gps, cams, targets, lidar_pts = load_kitti_sequence(config)
        print("Real KITTI data loaded.")
    except Exception as e:
        print(f"KITTI load failed: {e}\nUsing synthetic data.")
        data = synthetic_data.generate_sequence(dur=120)
        imus = np.column_stack([data['imu_acc'], data['imu_gyro']])
        cams = data['cam_imgs']
        targets = data['targets']
        lidar_pts = data['lidar_pts']

    # Convert to tensors
    imu_t = torch.tensor(imus, dtype=torch.float32).unsqueeze(1)
    lidar_t = torch.stack([torch.tensor(p, dtype=torch.float32) for p in lidar_pts])
    cam_t = torch.stack([torch.tensor(c, dtype=torch.float32).permute(2,0,1) for c in cams])
    target_t = torch.tensor(targets, dtype=torch.float32)

    dataset = TensorDataset(imu_t, lidar_t, cam_t, target_t)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = FusionModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    print("Training...")
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        for imu_b, lidar_b, cam_b, target_b in loader:
            imu_b = imu_b.to(device)
            lidar_b = lidar_b.to(device)
            cam_b = cam_b.to(device)
            target_b = target_b.to(device)
            pred = model(imu_b, lidar_b, cam_b)
            loss = torch.nn.MSELoss()(pred, target_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{config.num_epochs} Loss: {total_loss/len(loader):.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{config.model_type}_model.pth")
    print(f"Model saved as models/{config.model_type}_model.pth")

    model.eval()
    with torch.no_grad():
        pred_full = model(imu_t.to(device), lidar_t.to(device), cam_t.to(device)).cpu().numpy()
    ate = ATE(pred_full[:,:2], targets[:,:2])
    rpe = RPE(pred_full[:,:2], targets[:,:2])
    print(f"Training set ATE: {ate:.3f} m, RPE: {rpe:.3f} m")

if __name__ == "__main__":
    main()