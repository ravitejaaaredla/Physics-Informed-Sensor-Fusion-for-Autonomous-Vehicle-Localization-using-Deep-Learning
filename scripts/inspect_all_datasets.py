import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from src.data.kitti_dataset import KITTIRawDataset
from src.models.fusion_pinn import FusionPINN


def visualize_sample(camera, lidar_bev, dataset_name, output_dir):
    """Save a side-by-side visualization of Camera and LiDAR BEV."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Camera: convert from (3,128,416) to (128,416,3) for display
    cam_img = np.transpose(camera, (1, 2, 0))
    cam_img = (cam_img - cam_img.min()) / (cam_img.max() - cam_img.min() + 1e-6)
    axes[0].imshow(cam_img)
    axes[0].set_title(f"Camera - {dataset_name}")
    axes[0].axis('off')

    # LiDAR BEV: convert from (3,256,256) to (256,256,3)
    bev_img = np.transpose(lidar_bev, (1, 2, 0))
    bev_img = (bev_img - bev_img.min()) / (bev_img.max() - bev_img.min() + 1e-6)
    axes[1].imshow(bev_img)
    axes[1].set_title(f"LiDAR BEV - {dataset_name}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name}.png", dpi=150)
    plt.close()


def inspect_all_datasets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model (uses random weights if checkpoint not found, or loads pre-trained if available)
    model = FusionPINN(Config).to(device)
    model.eval()

    # Create output directory for visualizations
    vis_dir = PROJECT_ROOT / "runs" / "pre_train_inspect"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Collect all dataset paths with labels
    dataset_info = []

    # Training drives
    for i, d in enumerate(Config.train_sequence_dirs):
        dataset_info.append((d, f"train_{i}_drive{Config.train_drive_ids[i]}"))

    # Validation drives
    for i, d in enumerate(Config.val_sequence_dirs):
        dataset_info.append((d, f"val_{i}_drive{Config.val_drive_ids[i]}"))

    # Test drive
    dataset_info.append((Config.test_sequence_dir, f"test_drive{Config.test_drive_id}"))

    print("\n" + "=" * 80)
    print("INSPECTING ALL DATASETS")
    print("=" * 80)

    for path, label in dataset_info:
        print(f"\n{'='*60}")
        print(f"Dataset: {label}")
        print(f"Path: {path}")

        try:
            # Load only 1 frame from this dataset
            dataset = KITTIRawDataset([path], max_frames=1)
            sample = dataset[0]

            # Extract raw data
            camera = sample["camera"]           # (3,128,416)
            lidar_bev = sample["lidar_bev"]     # (3,256,256)
            imu = sample["imu"]                 # (6,)
            aux = sample["aux"]

            # Convert to batch and move to device
            camera_batch = camera.unsqueeze(0).to(device)
            lidar_batch = lidar_bev.unsqueeze(0).to(device)
            imu_batch = imu.unsqueeze(0).to(device)

            # ========== EXTRACT FEATURES USING THE CNN MODULES ==========
            with torch.no_grad():
                # 1. Camera CNN
                camera_feat = model.camera_cnn(camera_batch)
                print(f"  Camera CNN output shape: {camera_feat.shape}")  # Expected: (1, 64)

                # 2. LiDAR CNN
                lidar_feat = model.lidar_cnn(lidar_batch)
                print(f"  LiDAR CNN output shape: {lidar_feat.shape}")   # Expected: (1, 64)

                # 3. IMU Encoder
                imu_feat = model.imu_encoder(imu_batch)
                print(f"  IMU Encoder output shape: {imu_feat.shape}")   # Expected: (1, 128)

                # 4. Fused feature vector
                fused = torch.cat([camera_feat, lidar_feat, imu_feat], dim=1)
                print(f"  Fused feature vector shape: {fused.shape}")    # Expected: (1, 256)

                # 5. Full forward pass (PINN predicts motion)
                predicted_motion = model(camera_batch, lidar_batch, imu_batch)
                print(f"  PINN output (motion) shape: {predicted_motion.shape}")  # Expected: (1, 4)

            # Print a few ground truth values for context
            print(f"  Speed: {aux['speed_now']:.3f} m/s")
            print(f"  Yaw: {aux['yaw_now']:.4f} rad")
            print(f"  Position (x, y): ({aux['x_now']:.4f}, {aux['y_now']:.4f})")

            # Save visualization
            visualize_sample(camera.numpy(), lidar_bev.numpy(), label, vis_dir)

        except FileNotFoundError as e:
            print(f"  ❌ ERROR: {e}")
        except Exception as e:
            print(f"  ❌ UNEXPECTED ERROR: {e}")

    print("\n" + "=" * 80)
    print(f"VISUALIZATIONS SAVED TO: {vis_dir}")
    print("=" * 80)
    print("\n✅ Inspection complete! Check the output above for shape mismatches.")
    print("   If all shapes match, your pipeline is ready for training.")


if __name__ == "__main__":
    inspect_all_datasets()