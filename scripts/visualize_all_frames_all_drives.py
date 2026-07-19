import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from src.data.kitti_dataset import KITTIRawDataset


def visualize_single_frame(camera, lidar_bev, frame_idx, drive_label, output_dir):
    """Save a single frame visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Camera: (3,128,416) -> (128,416,3)
    cam_img = np.transpose(camera, (1, 2, 0))
    cam_img = (cam_img - cam_img.min()) / (cam_img.max() - cam_img.min() + 1e-6)
    axes[0].imshow(cam_img)
    axes[0].set_title(f"Camera - {drive_label} - Frame {frame_idx}")
    axes[0].axis('off')

    # LiDAR BEV: (3,256,256) -> (256,256,3)
    bev_img = np.transpose(lidar_bev, (1, 2, 0))
    bev_img = (bev_img - bev_img.min()) / (bev_img.max() - bev_img.min() + 1e-6)
    axes[1].imshow(bev_img)
    axes[1].set_title(f"LiDAR BEV - {drive_label} - Frame {frame_idx}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / f"frame_{frame_idx:04d}.png", dpi=100)
    plt.close()


def visualize_all_drives():
    """Loop through all drives and save every frame."""

    # Create a base output directory
    base_output = PROJECT_ROOT / "runs" / "full_visualization"
    base_output.mkdir(parents=True, exist_ok=True)

    # Collect all drives with labels
    drive_info = []

    # Training drives
    for i, d in enumerate(Config.train_sequence_dirs):
        drive_info.append((d, f"train_{Config.train_drive_ids[i]}"))

    # Validation drives
    for i, d in enumerate(Config.val_sequence_dirs):
        drive_info.append((d, f"val_{Config.val_drive_ids[i]}"))

    # Test drive
    drive_info.append((Config.test_sequence_dir, f"test_{Config.test_drive_id}"))

    print("=" * 80)
    print("VISUALIZING ALL FRAMES FOR ALL DRIVES")
    print("=" * 80)

    total_frames = 0

    for drive_path, label in drive_info:
        print(f"\n📁 Processing: {label}")

        # Load the entire drive (no max_frames limit)
        dataset = KITTIRawDataset([drive_path], max_frames=None)
        num_frames = len(dataset)
        print(f"   Total frames: {num_frames}")

        # Skip if empty
        if num_frames == 0:
            print("   ⚠️ Skipping (0 frames)")
            continue

        # Create subfolder for this drive
        drive_output = base_output / label
        drive_output.mkdir(parents=True, exist_ok=True)

        # Optionally, skip every 2nd/3rd frame to save disk space.
        # If you want ALL frames, set step = 1.
        step = 1
        print(f"   Saving every {step} frame(s)...")

        for idx in range(0, num_frames, step):
            sample = dataset[idx]
            camera = sample["camera"].numpy()
            lidar_bev = sample["lidar_bev"].numpy()

            visualize_single_frame(camera, lidar_bev, idx, label, drive_output)

            if idx % 50 == 0:
                print(f"      ... saved up to frame {idx}")

        total_frames += num_frames
        print(f"   ✅ Saved to: {drive_output}")

    print("\n" + "=" * 80)
    print(f"✅ COMPLETE! Total frames processed: {total_frames}")
    print(f"   All visualizations saved in: {base_output}")
    print("   Open this folder in Finder: open " + str(base_output))
    print("=" * 80)


if __name__ == "__main__":
    visualize_all_drives()