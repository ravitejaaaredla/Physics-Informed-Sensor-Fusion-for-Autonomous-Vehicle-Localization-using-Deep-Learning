# pinn_ekf.py

import os
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


class PINNIMUEKF:
    """
    EKF state:
        x = [px, py, v, yaw]

    Prediction:
        PINN predicts body-frame displacement:
            [dx_body, dy_body, dv, dyaw]

        We use:
            dx_body, dy_body, dv from PINN
            yaw update from IMU gyro_z

    Correction:
        sparse GNSS/OXTS position correction:
            z = [px, py]
    """

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        self.dt = getattr(config, "dt", 0.1)

        # State: [x, y, v, yaw]
        self.x = np.zeros(4, dtype=np.float32)

        # Covariance
        self.P = np.eye(4, dtype=np.float32)

        # Process noise
        q_pos = getattr(config, "process_noise_pos", 0.5)
        q_vel = getattr(config, "process_noise_vel", 0.5)
        q_yaw = getattr(config, "process_noise_yaw", 0.1)

        self.Q = np.diag(
            [
                q_pos ** 2,
                q_pos ** 2,
                q_vel ** 2,
                q_yaw ** 2,
            ]
        ).astype(np.float32)

        # GNSS/OXTS measurement noise
        gnss_noise = getattr(config, "gnss_noise_std", 1.0)

        self.R = np.diag(
            [
                gnss_noise ** 2,
                gnss_noise ** 2,
            ]
        ).astype(np.float32)

    def initialize(self, initial_state):
        """
        initial_state = [x, y, v, yaw]
        """
        self.x[:] = initial_state.astype(np.float32)

        self.P = np.diag(
            [
                1.0,
                1.0,
                0.5,
                0.1,
            ]
        ).astype(np.float32)

    def predict_pinn_delta(self, imu_win, lidar, cam):
        self.model.eval()

        imu_t = torch.tensor(imu_win, dtype=torch.float32).unsqueeze(0).to(self.device)
        lidar_t = torch.tensor(lidar, dtype=torch.float32).unsqueeze(0).to(self.device)
        cam_t = (
            torch.tensor(cam, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            delta = self.model(imu_t, lidar_t, cam_t).cpu().numpy()[0]

        return delta.astype(np.float32)

    def predict(self, imu_win, lidar, cam):
        """
        Prediction step:
            - translation from PINN
            - yaw from IMU gyro_z
        """
        delta = self.predict_pinn_delta(imu_win, lidar, cam)

        dx_body = float(delta[0])
        dy_body = float(delta[1])
        dv = float(delta[2])

        # Use IMU yaw-rate instead of learned dyaw
        gyro_z = float(imu_win[-1, 5])
        dyaw_imu = gyro_z * self.dt

        px, py, v, yaw = self.x

        c = np.cos(yaw)
        s = np.sin(yaw)

        # Body-frame displacement to world-frame displacement
        dx_world = c * dx_body - s * dy_body
        dy_world = s * dx_body + c * dy_body

        # State prediction
        px_new = px + dx_world
        py_new = py + dy_world
        v_new = max(0.0, v + dv)
        yaw_new = wrap_angle(yaw + dyaw_imu)

        self.x = np.array(
            [
                px_new,
                py_new,
                v_new,
                yaw_new,
            ],
            dtype=np.float32,
        )

        # Approximate Jacobian
        F = np.eye(4, dtype=np.float32)
        F[0, 3] = -s * dx_body - c * dy_body
        F[1, 3] = c * dx_body - s * dy_body

        self.P = F @ self.P @ F.T + self.Q

    def update_gnss(self, z_xy):
        """
        GNSS/OXTS position update.
        z_xy = [x, y]
        """
        z = np.asarray(z_xy, dtype=np.float32)

        H = np.zeros((2, 4), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[3] = wrap_angle(self.x[3])

        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ H) @ self.P


def run_ekf(
    model,
    states,
    lidar_pts,
    cam_imgs,
    imu_data,
    config,
    device,
    correction_interval=10,
    dropout_prob=0.0,
    random_seed=42,
):
    """
    Runs PINN + IMU yaw + EKF + sparse GNSS/OXTS correction.

    correction_interval:
        Apply GNSS/OXTS correction every N frames.

    dropout_prob:
        Probability of skipping a correction event.
    """
    rng = np.random.default_rng(random_seed)

    ekf = PINNIMUEKF(model, config, device)
    ekf.initialize(states[0])

    estimates = []
    estimates.append(ekf.x[:2].copy())

    correction_used = 0
    correction_skipped = 0

    for end in range(config.window_size - 1, len(states) - 1):
        start = end - config.window_size + 1

        imu_win = imu_data[start:end + 1]
        lidar = lidar_pts[end]
        cam = cam_imgs[end]

        ekf.predict(imu_win, lidar, cam)

        should_correct = (end % correction_interval == 0)

        if should_correct:
            if rng.random() >= dropout_prob:
                z_xy = states[end + 1, :2]
                ekf.update_gnss(z_xy)
                correction_used += 1
            else:
                correction_skipped += 1

        estimates.append(ekf.x[:2].copy())

    estimates = np.asarray(estimates, dtype=np.float32)
    gt_xy = states[: len(estimates), :2]

    return estimates, gt_xy, correction_used, correction_skipped


def plot_trajectory(gt_xy, pred_xy, title, save_path):
    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(gt_xy[:, 0], gt_xy[:, 1], "k-", linewidth=2.5, label="Ground Truth / OXTS")
    plt.plot(pred_xy[:, 0], pred_xy[:, 1], linewidth=2.0, label="PINN + IMU yaw + EKF")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


def plot_error(gt_xy, pred_xy, title, save_path):
    os.makedirs("results", exist_ok=True)

    n = min(len(gt_xy), len(pred_xy))
    error = np.linalg.norm(pred_xy[:n] - gt_xy[:n], axis=1)

    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(n), error, linewidth=1.5)
    plt.xlabel("Frame")
    plt.ylabel("Position Error (m)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


def evaluate_and_print(name, pred_xy, gt_xy, used, skipped):
    result_ate = ate(pred_xy, gt_xy)
    result_rpe = rpe(pred_xy, gt_xy)

    error = np.linalg.norm(pred_xy - gt_xy, axis=1)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"ATE: {result_ate:.3f} m")
    print(f"RPE: {result_rpe:.3f} m")
    print(f"Mean error: {np.mean(error):.3f} m")
    print(f"Median error: {np.median(error):.3f} m")
    print(f"Max error: {np.max(error):.3f} m")
    print(f"95th percentile error: {np.percentile(error, 95):.3f} m")
    print(f"GNSS/OXTS corrections used: {used}")
    print(f"GNSS/OXTS corrections skipped: {skipped}")

    return {
        "method": name,
        "ATE_m": result_ate,
        "RPE_m": result_rpe,
        "Mean_m": float(np.mean(error)),
        "Median_m": float(np.median(error)),
        "Max_m": float(np.max(error)),
        "P95_m": float(np.percentile(error, 95)),
        "Corrections_used": used,
        "Corrections_skipped": skipped,
    }


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

    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------
    # Main EKF localization experiment
    # ------------------------------------------------------------
    correction_interval = 10

    pred_xy, gt_xy, used, skipped = run_ekf(
        model=model,
        states=states,
        lidar_pts=lidar_pts,
        cam_imgs=cam_imgs,
        imu_data=imu_data,
        config=config,
        device=device,
        correction_interval=correction_interval,
        dropout_prob=0.0,
        random_seed=getattr(config, "seed", 42),
    )

    main_result = evaluate_and_print(
        name=f"PINN + IMU yaw + EKF/OXTS correction every {correction_interval} frames",
        pred_xy=pred_xy,
        gt_xy=gt_xy,
        used=used,
        skipped=skipped,
    )

    np.save("results/pinn_imu_ekf_pred_xy.npy", pred_xy)
    np.save("results/pinn_imu_ekf_gt_xy.npy", gt_xy)

    plot_trajectory(
        gt_xy,
        pred_xy,
        title=f"PINN + IMU Yaw + EKF/OXTS Correction Every {correction_interval} Frames",
        save_path="results/pinn_imu_ekf_trajectory.png",
    )

    plot_error(
        gt_xy,
        pred_xy,
        title="PINN + IMU Yaw + EKF Position Error",
        save_path="results/pinn_imu_ekf_error_over_time.png",
    )

    # ------------------------------------------------------------
    # GNSS/OXTS dropout robustness experiment
    # ------------------------------------------------------------
    dropout_probs = [0.0, 0.25, 0.5, 0.75, 1.0]
    robustness_rows = [main_result]

    for p in dropout_probs:
        pred_xy_d, gt_xy_d, used_d, skipped_d = run_ekf(
            model=model,
            states=states,
            lidar_pts=lidar_pts,
            cam_imgs=cam_imgs,
            imu_data=imu_data,
            config=config,
            device=device,
            correction_interval=correction_interval,
            dropout_prob=p,
            random_seed=getattr(config, "seed", 42),
        )

        row = evaluate_and_print(
            name=f"GNSS/OXTS dropout probability {p:.2f}",
            pred_xy=pred_xy_d,
            gt_xy=gt_xy_d,
            used=used_d,
            skipped=skipped_d,
        )

        row["dropout_probability"] = p
        robustness_rows.append(row)

    # Save CSV manually without pandas dependency
    csv_path = "results/gnss_dropout_ekf_results.csv"

    keys = [
        "method",
        "dropout_probability",
        "ATE_m",
        "RPE_m",
        "Mean_m",
        "Median_m",
        "Max_m",
        "P95_m",
        "Corrections_used",
        "Corrections_skipped",
    ]

    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")

        for row in robustness_rows:
            values = []
            for key in keys:
                values.append(str(row.get(key, "")))
            f.write(",".join(values) + "\n")

    print(f"\nSaved robustness results to: {csv_path}")


if __name__ == "__main__":
    main()