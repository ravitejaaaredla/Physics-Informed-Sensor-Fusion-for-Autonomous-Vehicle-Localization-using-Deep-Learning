import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from src.data.kitti_dataset import KITTIRawDataset
from src.models.fusion_pinn import FusionPINN


def show(name, tensor):
    print("\n" + name)
    print("-" * 80)
    print("shape:", tuple(tensor.shape))
    print("min:", float(tensor.min()))
    print("max:", float(tensor.max()))
    print("mean:", float(tensor.mean()))
    flat = tensor.detach().cpu().flatten()
    print("first 10 values:", flat[:10])


def main():
    print("=" * 80)
    print("01 - INSPECT KITTI DATA AND CNN FEATURES")
    print("=" * 80)

    dataset = KITTIRawDataset([Config.test_sequence_dir], max_frames=Config.max_frames)

    print("Test drive:", Config.test_sequence_dir)
    print("Number of samples:", len(dataset))

    sample = dataset[0]

    camera = sample["camera"]
    lidar_bev = sample["lidar_bev"]
    imu = sample["imu"]
    target = sample["target"]
    aux = sample["aux"]

    print("\nFrame:", aux["frame_idx"])
    print("Image:", aux["image_path"])
    print("LiDAR:", aux["lidar_path"])
    print("OXTS :", aux["oxts_path"])

    show("Camera input [3,128,416]", camera)
    show("LiDAR BEV input [3,256,256]", lidar_bev)
    show("IMU input [6]", imu)
    show("Ground truth target [4]", target)

    model = FusionPINN(Config)
    model.eval()

    camera_b = camera.unsqueeze(0)
    lidar_b = lidar_bev.unsqueeze(0)
    imu_b = imu.unsqueeze(0)

    with torch.no_grad():
        out = model.forward_with_features(camera_b, lidar_b, imu_b)

    show("Camera CNN feature f_cam [1,64]", out["f_cam"])
    show("LiDAR CNN feature f_lid [1,64]", out["f_lid"])
    show("IMU feature f_imu [1,128]", out["f_imu"])
    show("Fused PINN input z [1,256]", out["z"])
    show("Predicted motion [1,4]", out["predicted_motion"])

    print("\nMeaning:")
    print("Camera CNN gives 64 values.")
    print("LiDAR CNN gives 64 values.")
    print("IMU encoder gives 128 values.")
    print("Total values fed to PINN/fusion network = 64 + 64 + 128 = 256.")
    print("Output = [dx_body, dy_body, dv, dyaw].")


if __name__ == "__main__":
    main()
