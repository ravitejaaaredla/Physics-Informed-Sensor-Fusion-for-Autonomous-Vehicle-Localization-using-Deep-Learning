import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from src.data.kitti_dataset import KITTIRawDataset


def diagnose_drive(drive_path, label):
    print(f"\n{'-'*60}")
    print(f"Diagnosing: {label}")
    print(f"Path: {drive_path}")

    # Check if the folder exists
    if not drive_path.exists():
        print("  ❌ Folder does NOT exist!")
        return

    # Check for required subfolders
    required = ["image_02/data", "velodyne_points/data", "oxts/data"]
    missing = []
    for sub in required:
        sub_path = drive_path / sub
        if not sub_path.exists():
            missing.append(sub)
        else:
            # Count files
            files = list(sub_path.glob("*"))
            print(f"  ✅ {sub}: {len(files)} files")
    if missing:
        print(f"  ❌ Missing subfolders: {missing}")
        return

    # Try to load the dataset
    try:
        dataset = KITTIRawDataset([drive_path], max_frames=5)  # just load 5 to test
        print(f"  ✅ Dataset loaded with {len(dataset)} samples (max_frames=5)")
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"     Sample keys: {sample.keys()}")
            print(f"     Camera shape: {sample['camera'].shape}")
            print(f"     LiDAR BEV shape: {sample['lidar_bev'].shape}")
            print(f"     IMU shape: {sample['imu'].shape}")
            print(f"     Speed: {sample['aux']['speed_now']:.3f} m/s")
        else:
            print("  ⚠️ Dataset has 0 samples. Check file naming/numbering.")
    except Exception as e:
        print(f"  ❌ Failed to load dataset: {e}")


def main():
    print("=" * 80)
    print("DATASET DIAGNOSTIC TOOL")
    print("=" * 80)

    # Training drives
    for i, d in enumerate(Config.train_sequence_dirs):
        diagnose_drive(d, f"Train Drive {Config.train_drive_ids[i]}")

    # Validation drives
    for i, d in enumerate(Config.val_sequence_dirs):
        diagnose_drive(d, f"Val Drive {Config.val_drive_ids[i]}")

    # Test drive
    diagnose_drive(Config.test_sequence_dir, f"Test Drive {Config.test_drive_id}")

    print("\n" + "=" * 80)
    print("Diagnostic complete. If any dataset has 0 samples, check the folder structure.")
    print("Expected: image_02/data/*.png, velodyne_points/data/*.bin, oxts/data/*.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()