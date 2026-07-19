import sys
from pathlib import Path
import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from src.data.kitti_dataset import KITTIRawDataset
from src.localization.ekf import FourStateEKF
from src.localization.coordinates import xy_to_latlon
from src.evaluation.metrics import compute_ate, compute_rpe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_frame", type=int, default=200)
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--gnss_denied_start", type=int, default=200)
    parser.add_argument("--gnss_denied_len", type=int, default=50)
    args = parser.parse_args()

    print("=" * 80)
    print("EVALUATION WITH GROUND TRUTH MOTION (no model)")
    print("=" * 80)

    dataset = KITTIRawDataset([Config.test_sequence_dir], max_frames=Config.max_frames)
    print(f"Test samples: {len(dataset)}")

    start = args.start_frame
    end = min(start + args.num_frames, len(dataset))

    first_sample = dataset[start]
    aux0 = first_sample["aux"]
    origin_lat = aux0["lat_now"]
    origin_lon = aux0["lon_now"]

    ekf = FourStateEKF(Config)
    ekf.initialize(
        x=aux0["x_now"],
        y=aux0["y_now"],
        velocity=aux0["speed_now"],
        yaw=aux0["yaw_now"],
    )

    predicted_xy = []
    ground_truth_xy = []

    for idx in range(start, end):
        sample = dataset[idx]
        aux = sample["aux"]

        # ===== USE GROUND TRUTH MOTION =====
        # Ground truth motion is in the target, but we have it in aux.
        # We can compute dx, dy, dv from consecutive frames.
        if idx == start:
            dx_gt = 0.0
            dy_gt = 0.0
            dv_gt = 0.0
        else:
            prev_aux = dataset[idx - 1]["aux"]
            # Global position difference
            dx_world = aux["x_now"] - prev_aux["x_now"]
            dy_world = aux["y_now"] - prev_aux["y_now"]
            # Rotate to body frame using previous yaw
            yaw_prev = prev_aux["yaw_now"]
            dx_body = np.cos(yaw_prev) * dx_world + np.sin(yaw_prev) * dy_world
            dy_body = -np.sin(yaw_prev) * dx_world + np.cos(yaw_prev) * dy_world
            dv_gt = aux["speed_now"] - prev_aux["speed_now"]
            dx_gt, dy_gt = dx_body, dy_body

        # Use IMU for dyaw
        imu = sample["imu"].numpy()
        imu_omega_z = imu[5] * Config.imu_gyro_scale   # apply the same scale
        dyaw_gt = imu_omega_z * 0.1

        # Feed to EKF
        state = ekf.predict(dx_gt, dy_gt, dv_gt, dyaw_gt)

        # GNSS update (optional, but we will deny it for consistency)
        use_gnss = not (args.gnss_denied_start <= idx < args.gnss_denied_start + args.gnss_denied_len)
        if use_gnss:
            state = ekf.update_gnss(aux["x_next"], aux["y_next"])

        pred_x, pred_y = state[0], state[1]
        gt_x, gt_y = aux["x_next"], aux["y_next"]

        predicted_xy.append([pred_x, pred_y])
        ground_truth_xy.append([gt_x, gt_y])

    predicted_xy = np.array(predicted_xy)
    ground_truth_xy = np.array(ground_truth_xy)

    ate = compute_ate(predicted_xy, ground_truth_xy)
    print(f"\nMean ATE with ground truth motion: {np.mean(ate):.3f} m")
    print(f"Max ATE: {np.max(ate):.3f} m")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(ground_truth_xy[:, 0], ground_truth_xy[:, 1], 'b-', label='Ground Truth')
    plt.plot(predicted_xy[:, 0], predicted_xy[:, 1], 'r--', label='EKF with GT motion')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.title('EKF with Ground Truth Motion (IMU dyaw)')
    output_path = PROJECT_ROOT / "runs" / "gt_motion_eval.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()