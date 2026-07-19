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


def imu_ablation_test():
    device = torch.device("cpu")
    model = FusionPINN(Config).to(device)
    model.load_state_dict(torch.load(Config.checkpoint_dir / "model_best.pth", map_location=device))
    model.eval()

    dataset = KITTIRawDataset([Config.test_sequence_dir], max_frames=Config.max_frames)

    target_mean = np.array(Config.target_mean)
    target_std = np.array(Config.target_std)

    frames = list(range(200, 260))
    dyaw_gt = []
    dyaw_with_imu = []
    dyaw_zero_imu = []

    for idx in frames:
        sample = dataset[idx]
        camera = sample["camera"].unsqueeze(0).to(device)
        lidar = sample["lidar_bev"].unsqueeze(0).to(device)
        imu_real = sample["imu"].unsqueeze(0).to(device)
        imu_zero = torch.zeros_like(imu_real)  # Simulate broken IMU

        target = sample["target"].numpy()
        target_phys = (target * target_std) + target_mean
        dyaw_gt.append(target_phys[3])

        # Prediction with real IMU
        with torch.no_grad():
            pred_real = model(camera, lidar, imu_real).squeeze().cpu().numpy()
            pred_real_phys = (pred_real * target_std) + target_mean
            dyaw_with_imu.append(pred_real_phys[3])

        # Prediction with zero IMU
        with torch.no_grad():
            pred_zero = model(camera, lidar, imu_zero).squeeze().cpu().numpy()
            pred_zero_phys = (pred_zero * target_std) + target_mean
            dyaw_zero_imu.append(pred_zero_phys[3])

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(frames, dyaw_gt, 'b-', label='Ground Truth dyaw', linewidth=2)
    plt.plot(frames, dyaw_with_imu, 'r--', label='Prediction (with real IMU)', linewidth=2)
    plt.plot(frames, dyaw_zero_imu, 'g:', label='Prediction (with zero IMU)', linewidth=2)
    plt.xlabel('Frame')
    plt.ylabel('dyaw [rad]')
    plt.title('IMU Ablation Test: Does the Model Use IMU for Turning?')
    plt.legend()
    plt.grid(True)
    plt.axvline(x=230, color='black', linestyle='--', alpha=0.5, label='GNSS Denied Start')

    output_path = PROJECT_ROOT / "runs" / "eval" / "gnss_denied" / "imu_ablation.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved IMU ablation plot to: {output_path}")

    # Calculate differences
    diff_real = np.mean(np.abs(np.array(dyaw_gt) - np.array(dyaw_with_imu)))
    diff_zero = np.mean(np.abs(np.array(dyaw_gt) - np.array(dyaw_zero_imu)))
    print(f"\nAverage dyaw error with real IMU: {diff_real:.6f} rad")
    print(f"Average dyaw error with zero IMU: {diff_zero:.6f} rad")
    if abs(diff_real - diff_zero) < 0.0001:
        print("\n🚨 The model's predictions are IDENTICAL with or without IMU.")
        print("   This means the model is completely IGNORING the IMU data!")
    else:
        print("\n✅ The model IS using the IMU data, but not enough to correct the turn.")


if __name__ == "__main__":
    imu_ablation_test()