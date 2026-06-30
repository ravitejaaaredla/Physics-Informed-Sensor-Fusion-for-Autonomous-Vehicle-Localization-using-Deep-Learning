# evaluate_baselines.py

import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import Config
from src.models import FusionModel
from src.raw_kitti_utils import load_raw_sequence


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


def summarize(method, pred_xy, gt_xy):
    n = min(len(pred_xy), len(gt_xy))
    pred_xy = pred_xy[:n]
    gt_xy = gt_xy[:n]

    error = np.linalg.norm(pred_xy - gt_xy, axis=1)

    return {
        "method": method,
        "ATE_m": ate(pred_xy, gt_xy),
        "RPE_m": rpe(pred_xy, gt_xy),
        "Mean_m": float(np.mean(error)),
        "Median_m": float(np.median(error)),
        "Max_m": float(np.max(error)),
        "P95_m": float(np.percentile(error, 95)),
    }


def predict_delta(model, imu_win, lidar, cam, device):
    model.eval()

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

    return delta.astype(np.float32)


def integrate_pinn_learned_yaw(model, states, lidar_pts, cam_imgs, imu_data, config, device):
    pred_states = []

    x, y, v, yaw = states[0]
    pred_states.append([x, y, v, yaw])

    for end in range(config.window_size - 1, len(states) - 1):
        start = end - config.window_size + 1

        imu_win = imu_data[start:end + 1]
        lidar = lidar_pts[end]
        cam = cam_imgs[end]

        delta = predict_delta(model, imu_win, lidar, cam, device)

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


def integrate_pinn_imu_yaw(model, states, lidar_pts, cam_imgs, imu_data, config, device):
    pred_states = []

    x, y, v, yaw = states[0]
    pred_states.append([x, y, v, yaw])

    for end in range(config.window_size - 1, len(states) - 1):
        start = end - config.window_size + 1

        imu_win = imu_data[start:end + 1]
        lidar = lidar_pts[end]
        cam = cam_imgs[end]

        delta = predict_delta(model, imu_win, lidar, cam, device)

        dx_b, dy_b, dv, _ = delta

        gyro_z = imu_win[-1, 5]
        dyaw = gyro_z * config.dt

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


def integrate_oracle_yaw(model, states, lidar_pts, cam_imgs, imu_data, config, device):
    pred_states = []

    x, y, v, yaw = states[0]
    pred_states.append([x, y, v, yaw])

    for end in range(config.window_size - 1, len(states) - 1):
        start = end - config.window_size + 1

        imu_win = imu_data[start:end + 1]
        lidar = lidar_pts[end]
        cam = cam_imgs[end]

        delta = predict_delta(model, imu_win, lidar, cam, device)

        dx_b, dy_b, dv, _ = delta

        c = np.cos(yaw)
        s = np.sin(yaw)

        dx_w = c * dx_b - s * dy_b
        dy_w = s * dx_b + c * dy_b

        x = x + dx_w
        y = y + dy_w
        v = max(0.0, v + dv)

        # Diagnostic only: use ground-truth yaw
        yaw = states[len(pred_states), 3]

        pred_states.append([x, y, v, yaw])

    return np.asarray(pred_states, dtype=np.float32)


def ekf_only_baseline(states, imu_data, config, correction_interval=10):
    """
    Simple classical baseline:
    - Predict yaw using IMU gyro_z
    - Predict position using previous speed and yaw
    - Correct position using sparse OXTS/GNSS every correction_interval frames
    """
    x, y, v, yaw = states[0]

    P = np.diag([1.0, 1.0, 0.5, 0.1]).astype(np.float32)
    Q = np.diag([0.5 ** 2, 0.5 ** 2, 0.5 ** 2, 0.1 ** 2]).astype(np.float32)
    R = np.diag([1.0 ** 2, 1.0 ** 2]).astype(np.float32)

    estimates = []
    estimates.append([x, y])

    for k in range(config.window_size - 1, len(states) - 1):
        gyro_z = imu_data[k, 5]
        yaw = wrap_angle(yaw + gyro_z * config.dt)

        dx = v * np.cos(yaw) * config.dt
        dy = v * np.sin(yaw) * config.dt

        x = x + dx
        y = y + dy

        # speed update from ground-truth-like OXTS velocity proxy is not used.
        # Instead keep previous speed and let correction stabilize position.
        state = np.array([x, y, v, yaw], dtype=np.float32)

        F = np.eye(4, dtype=np.float32)
        F[0, 2] = np.cos(yaw) * config.dt
        F[1, 2] = np.sin(yaw) * config.dt
        F[0, 3] = -v * np.sin(yaw) * config.dt
        F[1, 3] = v * np.cos(yaw) * config.dt

        P = F @ P @ F.T + Q

        if k % correction_interval == 0:
            z = states[k + 1, :2].astype(np.float32)

            H = np.zeros((2, 4), dtype=np.float32)
            H[0, 0] = 1.0
            H[1, 1] = 1.0

            residual = z - H @ state
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)

            state = state + K @ residual
            P = (np.eye(4, dtype=np.float32) - K @ H) @ P

            x, y, v, yaw = state

        estimates.append([x, y])

    return np.asarray(estimates, dtype=np.float32)


class PINNIMUEKF:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.dt = config.dt

        self.x = np.zeros(4, dtype=np.float32)
        self.P = np.diag([1.0, 1.0, 0.5, 0.1]).astype(np.float32)

        self.Q = np.diag([0.5 ** 2, 0.5 ** 2, 0.5 ** 2, 0.1 ** 2]).astype(np.float32)
        self.R = np.diag([1.0 ** 2, 1.0 ** 2]).astype(np.float32)

    def initialize(self, initial_state):
        self.x[:] = initial_state.astype(np.float32)

    def predict(self, imu_win, lidar, cam):
        delta = predict_delta(self.model, imu_win, lidar, cam, self.device)

        dx_b, dy_b, dv, _ = delta

        gyro_z = imu_win[-1, 5]
        dyaw = gyro_z * self.dt

        px, py, v, yaw = self.x

        c = np.cos(yaw)
        s = np.sin(yaw)

        dx_w = c * dx_b - s * dy_b
        dy_w = s * dx_b + c * dy_b

        px = px + dx_w
        py = py + dy_w
        v = max(0.0, v + dv)
        yaw = wrap_angle(yaw + dyaw)

        self.x = np.array([px, py, v, yaw], dtype=np.float32)

        F = np.eye(4, dtype=np.float32)
        F[0, 3] = -s * dx_b - c * dy_b
        F[1, 3] = c * dx_b - s * dy_b

        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_xy):
        z = np.asarray(z_xy, dtype=np.float32)

        H = np.zeros((2, 4), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        residual = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ residual
        self.x[3] = wrap_angle(self.x[3])

        self.P = (np.eye(4, dtype=np.float32) - K @ H) @ self.P


def pinn_imu_ekf(model, states, lidar_pts, cam_imgs, imu_data, config, device, correction_interval=10):
    ekf = PINNIMUEKF(model, config, device)
    ekf.initialize(states[0])

    estimates = []
    estimates.append(ekf.x[:2].copy())

    for end in range(config.window_size - 1, len(states) - 1):
        start = end - config.window_size + 1

        imu_win = imu_data[start:end + 1]
        lidar = lidar_pts[end]
        cam = cam_imgs[end]

        ekf.predict(imu_win, lidar, cam)

        if end % correction_interval == 0:
            ekf.update(states[end + 1, :2])

        estimates.append(ekf.x[:2].copy())

    return np.asarray(estimates, dtype=np.float32)


def plot_comparison(gt_xy, trajectories):
    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(9, 7))
    plt.plot(gt_xy[:, 0], gt_xy[:, 1], "k-", linewidth=3.0, label="Ground Truth / OXTS")

    for name, xy in trajectories.items():
        n = min(len(xy), len(gt_xy))
        plt.plot(xy[:n, 0], xy[:n, 1], linewidth=1.8, label=name)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Baseline Comparison")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/baseline_comparison_trajectory.png", dpi=200)
    plt.show()


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    states, lidar_pts, cam_imgs, imu_data = load_raw_sequence(config.seq_dir, config)

    model_path = "models/pinn_delta_physics.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Run first: python train_physics_loss.py"
        )

    model = FusionModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    trajectories = {}

    # 1. Pure PINN learned yaw
    traj_learned = integrate_pinn_learned_yaw(
        model, states, lidar_pts, cam_imgs, imu_data, config, device
    )
    gt_xy = states[: len(traj_learned), :2]
    pred_xy = traj_learned[:, :2]
    results.append(summarize("Pure PINN learned yaw", pred_xy, gt_xy))
    trajectories["Pure PINN learned yaw"] = pred_xy

    # 2. PINN + IMU yaw
    traj_imu = integrate_pinn_imu_yaw(
        model, states, lidar_pts, cam_imgs, imu_data, config, device
    )
    gt_xy = states[: len(traj_imu), :2]
    pred_xy = traj_imu[:, :2]
    results.append(summarize("PINN + IMU yaw", pred_xy, gt_xy))
    trajectories["PINN + IMU yaw"] = pred_xy

    # 3. PINN + IMU yaw + EKF
    pred_ekf = pinn_imu_ekf(
        model, states, lidar_pts, cam_imgs, imu_data, config, device, correction_interval=10
    )
    gt_xy = states[: len(pred_ekf), :2]
    results.append(summarize("PINN + IMU yaw + EKF/OXTS", pred_ekf, gt_xy))
    trajectories["PINN + IMU yaw + EKF/OXTS"] = pred_ekf

    # 4. EKF-only baseline
    pred_ekf_only = ekf_only_baseline(states, imu_data, config, correction_interval=10)
    gt_xy = states[: len(pred_ekf_only), :2]
    results.append(summarize("EKF-only IMU + OXTS", pred_ekf_only, gt_xy))
    trajectories["EKF-only IMU + OXTS"] = pred_ekf_only

    # 5. Oracle yaw diagnostic
    traj_oracle = integrate_oracle_yaw(
        model, states, lidar_pts, cam_imgs, imu_data, config, device
    )
    gt_xy = states[: len(traj_oracle), :2]
    pred_xy = traj_oracle[:, :2]
    results.append(summarize("Oracle yaw diagnostic", pred_xy, gt_xy))
    trajectories["Oracle yaw diagnostic"] = pred_xy

    print("\nBaseline comparison")
    print("=" * 100)
    print(f"{'Method':35s} {'ATE':>10s} {'RPE':>10s} {'Mean':>10s} {'Median':>10s} {'Max':>10s} {'P95':>10s}")
    print("-" * 100)

    for r in results:
        print(
            f"{r['method']:35s} "
            f"{r['ATE_m']:10.3f} "
            f"{r['RPE_m']:10.3f} "
            f"{r['Mean_m']:10.3f} "
            f"{r['Median_m']:10.3f} "
            f"{r['Max_m']:10.3f} "
            f"{r['P95_m']:10.3f}"
        )

    os.makedirs("results", exist_ok=True)
    csv_path = "results/baseline_comparison_full.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "ATE_m",
                "RPE_m",
                "Mean_m",
                "Median_m",
                "Max_m",
                "P95_m",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    plot_comparison(states[: len(traj_learned), :2], trajectories)

    print(f"\nSaved results to: {csv_path}")
    print("Saved plot to: results/baseline_comparison_trajectory.png")


if __name__ == "__main__":
    main()