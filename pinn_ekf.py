import numpy as np
import torch
from config import Config
from src.kitti_loader import load_kitti_sequence
from src.models import FusionModel
from src.eval.metrics import ATE, RPE

class PINNEKF:
    def __init__(self, model, dt, device):
        self.model = model
        self.dt = dt
        self.device = device
        # State: [x, y, vx, vy, yaw]
        self.x = np.zeros(5)
        self.P = np.eye(5) * 1.0
        self.Q = np.diag([0.5, 0.5, 0.5, 0.5, 0.1]) ** 2
        self.R = np.diag([1.0, 1.0]) ** 2

    def predict_delta(self, imu_win, lidar, cam):
        with torch.no_grad():
            imu_t = torch.tensor(imu_win, dtype=torch.float32).unsqueeze(0).to(self.device)
            lidar_t = torch.tensor(lidar, dtype=torch.float32).unsqueeze(0).to(self.device)
            cam_t = torch.tensor(cam, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(self.device)
            delta = self.model(imu_t, lidar_t, cam_t).cpu().numpy()[0]
        return delta

    def predict(self, imu_win, lidar, cam):
        # State transition using PINN delta
        delta = self.predict_delta(imu_win, lidar, cam)
        dx, dy, dv, dyaw = delta
        yaw = self.x[4]
        self.x[0] += dx
        self.x[1] += dy
        self.x[2] += dv * np.cos(yaw)
        self.x[3] += dv * np.sin(yaw)
        self.x[4] += dyaw
        # Simplified covariance propagation (no Jacobian)
        self.P = self.P + self.Q

    def update_gnss(self, z):
        H = np.zeros((2,5))
        H[0,0] = 1
        H[1,1] = 1
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(5) - K @ H) @ self.P

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load data (8-return loader with normalisation)
    bevs, imus, gps, cams, targets_norm, lidar_pts, pos_mean, pos_std = load_kitti_sequence(config)

    # Load trained PINN model (ensure models/pinn_physics.pth exists)
    model = FusionModel(config).to(device)
    try:
        model.load_state_dict(torch.load("models/pinn_physics.pth", map_location=device))
        print("Loaded PINN model from models/pinn_physics.pth")
    except FileNotFoundError:
        print("Model not found. Please train the model first with train_physics_loss.py")
        return
    model.eval()

    ekf = PINNEKF(model, dt=0.1, device=device)
    window_size = 10
    est_positions_norm = []

    for i in range(len(imus) - window_size):
        imu_win = imus[i:i+window_size]
        lidar = lidar_pts[i+window_size-1]
        cam = cams[i+window_size-1]
        ekf.predict(imu_win, lidar, cam)
        if i % 10 == 0 and i//10 < len(gps):
            ekf.update_gnss(gps[i//10])
        est_positions_norm.append(ekf.x[:2].copy())

    est_positions_norm = np.array(est_positions_norm)
    # Denormalise predictions
    est_positions = est_positions_norm * pos_std + pos_mean
    # Reconstruct original targets from normalised ones
    targets_orig = targets_norm.copy()
    targets_orig[:,:2] = targets_norm[:,:2] * pos_std + pos_mean
    min_len = min(len(est_positions), len(targets_orig))
    ate = ATE(est_positions[:min_len], targets_orig[:min_len, :2])
    rpe = RPE(est_positions[:min_len], targets_orig[:min_len, :2])
    print(f"PINN+EKF hybrid: ATE = {ate:.3f} m, RPE = {rpe:.3f} m")

if __name__ == "__main__":
    main()