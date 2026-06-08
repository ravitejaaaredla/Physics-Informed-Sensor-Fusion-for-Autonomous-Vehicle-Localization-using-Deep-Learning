import matplotlib.pyplot as plt
import numpy as np
import torch
from config import Config
from src.kitti_loader import load_kitti_sequence
from src.models import FusionModel
from src.eval.metrics import ATE, RPE

config = Config()
bevs, imus, gps, cams, targets_norm, lidar_pts, pos_mean, pos_std = load_kitti_sequence(config)
model = FusionModel(config).to('cpu')
model.load_state_dict(torch.load('models/pinn_singleframe.pth', map_location='cpu'))
model.eval()
imu_t = torch.tensor(imus, dtype=torch.float32).unsqueeze(1)
lidar_t = torch.stack([torch.tensor(p, dtype=torch.float32) for p in lidar_pts])
cam_t = torch.stack([torch.tensor(c, dtype=torch.float32).permute(2,0,1) for c in cams])
with torch.no_grad():
    pred_norm = model(imu_t, lidar_t, cam_t).numpy()
pred = pred_norm.copy()
pred[:,:2] = pred_norm[:,:2] * pos_std + pos_mean
targets_orig = targets_norm.copy()
targets_orig[:,:2] = targets_norm[:,:2] * pos_std + pos_mean

error = np.linalg.norm(pred[:,:2] - targets_orig[:,:2], axis=1)
frames = np.arange(len(error))

plt.figure(figsize=(10,4))
plt.plot(frames, error, 'b-', linewidth=1)
plt.xlabel("Frame number")
plt.ylabel("Position error (m)")
plt.title("Absolute Trajectory Error per Frame")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/error_over_time.png", dpi=150)
plt.show()
print(f"Mean error: {np.mean(error):.2f} m, Max error: {np.max(error):.2f} m")
