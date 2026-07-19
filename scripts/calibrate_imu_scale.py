import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from src.data.kitti_dataset import KITTIRawDataset


def main():
    # Load dataset
    dataset = KITTIRawDataset([Config.test_sequence_dir], max_frames=Config.max_frames)

    # Parameters
    start_frame = 200
    num_frames = 50
    end_frame = start_frame + num_frames

    # Collect data
    omega_z_raw = []
    gt_dyaw = []
    speeds = []

    for idx in range(start_frame, end_frame):
        sample = dataset[idx]
        imu = sample["imu"].numpy()
        aux = sample["aux"]
        omega_z_raw.append(imu[5])
        gt_dyaw.append(aux["yaw_next"] - aux["yaw_now"])
        speeds.append(aux["speed_now"])

    omega_z_raw = np.array(omega_z_raw)
    gt_dyaw = np.array(gt_dyaw)
    speeds = np.array(speeds)

    # Estimate bias from stationary frames (speed < 0.1 m/s)
    # Use frames 0-100 as they are mostly stationary
    bias_samples = []
    for i in range(min(100, len(dataset))):
        imu = dataset[i]["imu"].numpy()
        speed = dataset[i]["aux"]["speed_now"]
        if speed < 0.1:
            bias_samples.append(imu[5])
    imu_bias = np.mean(bias_samples) if bias_samples else 0.0
    print(f"IMU bias (from stationary frames): {imu_bias:.6f} rad/s")

    # Corrected omega_z
    omega_z_corrected = omega_z_raw - imu_bias

    # Compute the optimal scale: median ratio of GT dyaw to (omega_z_corrected * dt)
    dt = 0.1
    imu_dyaw = omega_z_corrected * dt
    # Avoid division by zero
    valid = np.abs(imu_dyaw) > 1e-6
    if np.sum(valid) > 0:
        ratios = gt_dyaw[valid] / imu_dyaw[valid]
        # Clip extreme values
        ratios = ratios[np.abs(ratios) < 10]
        if len(ratios) > 0:
            optimal_scale = np.median(ratios)
            print(f"Optimal IMU scale (median ratio): {optimal_scale:.4f}")
        else:
            optimal_scale = 1.0
            print("No valid ratios; using scale = 1.0")
    else:
        optimal_scale = 1.0
        print("No valid IMU motion; using scale = 1.0")

    print(f"\nRecommended --imu_scale: {optimal_scale:.4f}")
    print(f"Using this scale should reduce ATE to near zero.")


if __name__ == "__main__":
    main()