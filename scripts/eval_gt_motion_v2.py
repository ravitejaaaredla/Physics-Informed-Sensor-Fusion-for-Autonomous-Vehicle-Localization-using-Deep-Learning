import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from src.data.kitti_dataset import KITTIRawDataset
from src.localization.ekf import FourStateEKF
from src.evaluation.metrics import compute_ate


def main():
    start_frame = 200
    num_frames = 50

    dataset = KITTIRawDataset([Config.test_sequence_dir], max_frames=Config.max_frames)
    print(f"Total frames: {len(dataset)}")

    # Initialize with start frame
    aux0 = dataset[start_frame]["aux"]
    ekf = FourStateEKF(Config)
    ekf.initialize(
        x=aux0["x_now"],
        y=aux0["y_now"],
        velocity=aux0["speed_now"],
        yaw=aux0["yaw_now"],
    )

    predicted_xy = []
    ground_truth_xy = []

    for idx in range(start_frame, start_frame + num_frames):
        sample = dataset[idx]
        aux = sample["aux"]

        # Ground truth position (use x_now, y_now directly)
        gt_x = aux["x_now"]
        gt_y = aux["y_now"]
        ground_truth_xy.append([gt_x, gt_y])

        # If not the last frame, compute motion to next frame
        if idx < start_frame + num_frames - 1:
            next_aux = dataset[idx + 1]["aux"]
            # Global displacement
            dx_world = next_aux["x_now"] - aux["x_now"]
            dy_world = next_aux["y_now"] - aux["y_now"]
            # Rotate to body frame using current yaw
            yaw = aux["yaw_now"]
            dx_body = np.cos(yaw) * dx_world + np.sin(yaw) * dy_world
            dy_body = -np.sin(yaw) * dx_world + np.cos(yaw) * dy_world
            dv = next_aux["speed_now"] - aux["speed_now"]
            # Use ground truth dyaw (from aux) instead of IMU
            dyaw = next_aux["yaw_now"] - aux["yaw_now"]
            # OR use IMU: we can test both
            # imu = sample["imu"].numpy()
            # imu_omega_z = imu[5] * Config.imu_gyro_scale
            # dyaw = imu_omega_z * 0.1
            ekf.predict(dx_body, dy_body, dv, dyaw)

        # Record EKF state (position after prediction)
        pred_x, pred_y = ekf.x[0], ekf.x[1]
        predicted_xy.append([pred_x, pred_y])

    predicted_xy = np.array(predicted_xy)
    ground_truth_xy = np.array(ground_truth_xy)

    # Compute ATE
    ate = compute_ate(predicted_xy, ground_truth_xy)
    print(f"Mean ATE: {np.mean(ate):.3f} m")
    print(f"Max ATE: {np.max(ate):.3f} m")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(ground_truth_xy[:, 0], ground_truth_xy[:, 1], 'b-', label='Ground Truth')
    plt.plot(predicted_xy[:, 0], predicted_xy[:, 1], 'r--', label='EKF with GT motion & GT dyaw')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.title('EKF with Ground Truth Motion and Ground Truth dyaw')
    out_path = PROJECT_ROOT / "runs" / "gt_motion_v2.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {out_path}")

if __name__ == "__main__":
    main()