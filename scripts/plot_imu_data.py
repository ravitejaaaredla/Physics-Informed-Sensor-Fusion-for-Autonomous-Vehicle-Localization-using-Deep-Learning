import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from src.data.kitti_dataset import KITTIRawDataset


def plot_imu_data():
    # Load dataset
    dataset = KITTIRawDataset([Config.test_sequence_dir], max_frames=Config.max_frames)
    print(f"Loaded {len(dataset)} samples from test drive.")

    # We'll analyze frames 150-260 (to see the transition from parked to moving and the turn)
    start_frame = 150
    end_frame = min(260, len(dataset))

    frames = []
    imu_data = []  # [ax, ay, az, wx, wy, wz]
    speeds = []
    yaws = []

    for idx in range(start_frame, end_frame):
        sample = dataset[idx]
        imu = sample["imu"].numpy()  # shape: (6,)
        aux = sample["aux"]

        frames.append(idx)
        imu_data.append(imu)
        speeds.append(aux["speed_now"])
        yaws.append(aux["yaw_now"])

    imu_data = np.array(imu_data)
    speeds = np.array(speeds)
    yaws = np.array(yaws)

    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Speed & Yaw
    ax1 = axes[0]
    ax1.plot(frames, speeds, 'b-', label='Speed (m/s)', linewidth=2)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Speed (m/s)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax1_2 = ax1.twinx()
    ax1_2.plot(frames, yaws, 'r-', label='Yaw (rad)', linewidth=2)
    ax1_2.set_ylabel('Yaw (rad)', color='r')
    ax1_2.tick_params(axis='y', labelcolor='r')
    ax1_2.legend(loc='upper right')
    ax1.set_title('Speed and Heading (Yaw)')

    # Plot 2: Acceleration (ax, ay, az)
    ax2 = axes[1]
    ax2.plot(frames, imu_data[:, 0], label='ax (forward accel)', linewidth=1.5)
    ax2.plot(frames, imu_data[:, 1], label='ay (lateral accel)', linewidth=1.5)
    ax2.plot(frames, imu_data[:, 2], label='az (vertical accel)', linewidth=1.5)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Acceleration (m/s²)')
    ax2.set_title('IMU Acceleration')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Angular Velocity (wx, wy, wz) - THIS IS THE TURNING SIGNAL!
    ax3 = axes[2]
    ax3.plot(frames, imu_data[:, 3], label='ωx (roll rate)', linewidth=1.5)
    ax3.plot(frames, imu_data[:, 4], label='ωy (pitch rate)', linewidth=1.5)
    ax3.plot(frames, imu_data[:, 5], label='ωz (yaw rate - TURNING!)', linewidth=2, color='red')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Angular Velocity (rad/s)')
    ax3.set_title('IMU Gyroscope (Angular Velocity) - ωz shows the turn!')
    ax3.legend()
    ax3.grid(True)

    # Add vertical line at frame 230 (start of GNSS denial)
    for ax in axes:
        ax.axvline(x=230, color='black', linestyle='--', alpha=0.5, label='GNSS Denied Start (Frame 230)')
        ax.axvline(x=259, color='black', linestyle=':', alpha=0.5, label='End of Denied Interval')

    plt.tight_layout()
    output_path = PROJECT_ROOT / "runs" / "eval" / "gnss_denied" / "imu_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved IMU analysis plot to: {output_path}")
    print(f"\nOpening the plot...")
    import os
    os.system(f"open {output_path}")


if __name__ == "__main__":
    plot_imu_data()