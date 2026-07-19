import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from src.data.kitti_dataset import read_oxts_file


def check_sync():
    print("=" * 80)
    print("SENSOR SYNCHRONIZATION CHECK")
    print("=" * 80)

    # 1. Check all training drives
    all_drives = Config.train_sequence_dirs + Config.val_sequence_dirs + [Config.test_sequence_dir]
    all_labels = Config.train_drive_ids + Config.val_drive_ids + [Config.test_drive_id]

    for drive_path, label in zip(all_drives, all_labels):
        print(f"\nDrive {label}: {drive_path}")

        image_dir = drive_path / "image_02" / "data"
        lidar_dir = drive_path / "velodyne_points" / "data"
        oxts_dir = drive_path / "oxts" / "data"

        image_files = sorted(image_dir.glob("*.png"))
        lidar_files = sorted(lidar_dir.glob("*.bin"))
        oxts_files = sorted(oxts_dir.glob("*.txt"))

        n_images = len(image_files)
        n_lidar = len(lidar_files)
        n_oxts = len(oxts_files)

        print(f"  Image files : {n_images}")
        print(f"  LiDAR files : {n_lidar}")
        print(f"  OXTS files  : {n_oxts}")

        if n_images == n_lidar == n_oxts:
            print("  ✅ All file counts match.")
        else:
            print("  ⚠️ File counts MISMATCH! Check the drive folder.")
            print(f"     Min = {min(n_images, n_lidar, n_oxts)}")

        # 2. Check OXTS timestamps (first 10 frames)
        print("\n  OXTS Timestamps (first 10 frames):")
        print("  Frame | GPS Time (s) | Δt (s)")
        print("  ------|--------------|--------")
        prev_time = None
        for i in range(min(10, n_oxts)):
            oxts = read_oxts_file(oxts_files[i])
            gps_time = oxts[0]  # KITTI OXTS index 0 is GPS time
            if prev_time is not None:
                dt = gps_time - prev_time
                print(f"  {i:5d} | {gps_time:12.6f} | {dt:7.4f}")
            else:
                print(f"  {i:5d} | {gps_time:12.6f} |   N/A")
            prev_time = gps_time

        # 3. Check if the dataset pairs files correctly for frame 0, 1, 2
        print("\n  File pairing for first 3 frames:")
        for i in range(min(3, n_images, n_lidar, n_oxts)):
            img_name = image_files[i].name
            lidar_name = lidar_files[i].name
            oxts_name = oxts_files[i].name
            print(f"  Frame {i}:")
            print(f"    Image : {img_name}")
            print(f"    LiDAR : {lidar_name}")
            print(f"    OXTS  : {oxts_name}")

        # 4. Check if the dataset returns the correct index for frame 200
        if n_oxts > 200:
            print(f"\n  Checking frame 200 (used for testing):")
            img_200 = image_files[200].name
            lidar_200 = lidar_files[200].name
            oxts_200 = oxts_files[200].name
            print(f"    Image : {img_200}")
            print(f"    LiDAR : {lidar_200}")
            print(f"    OXTS  : {oxts_200}")
        else:
            print(f"\n  ⚠️ Drive {label} has fewer than 200 frames.")

        print("-" * 60)

    # 5. Test the dataset loader directly for a few frames
    print("\n" + "=" * 80)
    print("DATASET LOADER TEST (first 5 frames of test drive)")
    print("=" * 80)

    from src.data.kitti_dataset import KITTIRawDataset

    dataset = KITTIRawDataset([Config.test_sequence_dir], max_frames=5)
    for i in range(len(dataset)):
        sample = dataset[i]
        aux = sample["aux"]
        imu = sample["imu"].numpy()
        print(f"\nFrame {i}:")
        print(f"  OXTS file  : {aux['oxts_path']}")
        print(f"  Image file : {aux['image_path']}")
        print(f"  LiDAR file : {aux['lidar_path']}")
        print(f"  Speed      : {aux['speed_now']:.4f} m/s")
        print(f"  Yaw        : {aux['yaw_now']:.6f} rad")
        print(f"  IMU omega_z: {imu[5]:.6f}")

    print("\n" + "=" * 80)
    print("✅ Check complete.")
    print("If all file counts match and timestamps are ~0.1s apart, your data is synced.")
    print("=" * 80)


if __name__ == "__main__":
    check_sync()