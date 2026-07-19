import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from src.data.kitti_dataset import KITTIRawDataset, read_oxts_file


def check_sync_and_plot():
    print("=" * 80)
    print("SYNCHRONIZATION CHECK & GROUND TRUTH PLOT")
    print("=" * 80)

    # 1. Load the test dataset
    dataset = KITTIRawDataset([Config.test_sequence_dir], max_frames=Config.max_frames)
    print(f"\nTest drive: {Config.test_sequence_dir}")
    print(f"Total frames in dataset: {len(dataset)}")

    # 2. Extract ground truth latitude, longitude from the dataset
    lats = []
    lons = []
    speeds = []
    yaws = []
    frame_indices = []

    for i in range(len(dataset)):
        sample = dataset[i]
        aux = sample["aux"]
        lats.append(aux["lat_now"])
        lons.append(aux["lon_now"])
        speeds.append(aux["speed_now"])
        yaws.append(aux["yaw_now"])
        frame_indices.append(i)

    lats = np.array(lats)
    lons = np.array(lons)
    speeds = np.array(speeds)

    # 3. Check OXTS timestamps for first 10 frames (sync check)
    print("\n--- OXTS Timestamps (first 10 frames) ---")
    oxts_files = sorted((Config.test_sequence_dir / "oxts" / "data").glob("*.txt"))
    prev_time = None
    for i in range(min(10, len(oxts_files))):
        oxts = read_oxts_file(oxts_files[i])
        gps_time = oxts[0]  # GPS time in seconds
        if prev_time is not None:
            dt = gps_time - prev_time
            print(f"Frame {i}: GPS time = {gps_time:.6f}, Δt = {dt:.4f} s")
        else:
            print(f"Frame {i}: GPS time = {gps_time:.6f}, Δt = N/A")
        prev_time = gps_time

    # 4. Count files in each sensor folder
    image_dir = Config.test_sequence_dir / "image_02" / "data"
    lidar_dir = Config.test_sequence_dir / "velodyne_points" / "data"
    oxts_dir = Config.test_sequence_dir / "oxts" / "data"

    n_img = len(list(image_dir.glob("*.png")))
    n_lidar = len(list(lidar_dir.glob("*.bin")))
    n_oxts = len(list(oxts_dir.glob("*.txt")))

    print(f"\n--- File Counts ---")
    print(f"Images : {n_img}")
    print(f"LiDAR  : {n_lidar}")
    print(f"OXTS   : {n_oxts}")
    if n_img == n_lidar == n_oxts:
        print("✅ All file counts match.")
    else:
        print("⚠️ File counts MISMATCH! Check drive folder.")

    # 5. Plot ground truth trajectory in latitude/longitude
    plt.figure(figsize=(10, 8))
    plt.plot(lons, lats, 'b-', linewidth=2, label='Ground Truth (OXTS)')
    plt.scatter(lons[0], lats[0], s=100, c='green', marker='o', label='Start')
    plt.scatter(lons[-1], lats[-1], s=100, c='red', marker='s', label='End')

    # Annotate start/end
    plt.annotate('Start', (lons[0], lats[0]), xytext=(5, 5), textcoords='offset points')
    plt.annotate('End', (lons[-1], lats[-1]), xytext=(5, -10), textcoords='offset points')

    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.title('Ground Truth Trajectory (Latitude / Longitude)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # 6. Try to overlay predicted trajectory from evaluation CSV if available
    csv_path = Config.eval_dir / "gnss_denied" / "trajectory_latlon.csv"
    if csv_path.exists():
        print(f"\nFound evaluation CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        pred_lat = df["predicted_latitude"].values
        pred_lon = df["predicted_longitude"].values
        # Plot only the predicted part (frames 200-250)
        plt.plot(pred_lon, pred_lat, 'r--', linewidth=2, label='Predicted (PINN+EKF)')
        # Mark denied interval if applicable
        denied = df[df["use_gnss"] == False]
        if len(denied) > 0:
            plt.plot(denied["predicted_longitude"], denied["predicted_latitude"],
                     'r.', markersize=3, label='GNSS Denied (predicted)')
    else:
        print("\nNo evaluation CSV found. Only ground truth plotted.")

    plt.tight_layout()
    out_path = PROJECT_ROOT / "runs" / "ground_truth_trajectory.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n✅ Trajectory plot saved to: {out_path}")
    print("   Open it with: open " + str(out_path))


if __name__ == "__main__":
    check_sync_and_plot()