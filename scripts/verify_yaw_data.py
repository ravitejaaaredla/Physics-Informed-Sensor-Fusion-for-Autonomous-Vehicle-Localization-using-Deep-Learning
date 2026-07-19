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


def main():
    print("=" * 80)
    print("YAW VERIFICATION TOOL")
    print("=" * 80)

    # 1. Check the EKF parameter being used
    print(f"\nConfig.process_noise_yaw = {Config.process_noise_yaw}")
    print(f"Config.imu_dyaw_scale   = {Config.imu_dyaw_scale}")
    print(f"Config.lambda_imu       = {Config.lambda_imu}")

    # 2. Load model and dataset
    device = torch.device("cpu")
    dataset = KITTIRawDataset([Config.test_sequence_dir], max_frames=Config.max_frames)
    model = FusionPINN(Config).to(device)
    model_path = Config.checkpoint_dir / "model_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model: {model_path}")

    target_mean = np.array(Config.target_mean)
    target_std = np.array(Config.target_std)

    # 3. Collect data for frames 200 to 250
    start_frame = 200
    end_frame = 250

    frames = []
    gt_dyaw = []
    model_dyaw = []
    imu_dyaw = []

    for idx in range(start_frame, end_frame):
        sample = dataset[idx]
        camera = sample["camera"].unsqueeze(0).to(device)
        lidar = sample["lidar_bev"].unsqueeze(0).to(device)
        imu = sample["imu"].unsqueeze(0).to(device)
        aux = sample["aux"]

        # Ground Truth
        gt = aux["yaw_next"] - aux["yaw_now"]

        # Model Prediction
        with torch.no_grad():
            pred = model(camera, lidar, imu)
        pred_norm = pred[0].cpu().numpy()
        pred_phys = (pred_norm * target_std) + target_mean
        model_dy = pred_phys[3]

        # IMU Prediction (raw gyro)
        imu_omega_z = imu[0, 5].item()
        imu_dy = imu_omega_z * 0.1  # dt = 0.1

        frames.append(idx)
        gt_dyaw.append(gt)
        model_dyaw.append(model_dy)
        imu_dyaw.append(imu_dy)

    # 4. Convert to numpy
    frames = np.array(frames)
    gt_dyaw = np.array(gt_dyaw)
    model_dyaw = np.array(model_dyaw)
    imu_dyaw = np.array(imu_dyaw)

    # 5. Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS (Frames 200-250)")
    print("-" * 80)
    print(f"Mean GT dyaw      : {np.mean(gt_dyaw):.6f} rad")
    print(f"Mean Model dyaw   : {np.mean(model_dyaw):.6f} rad")
    print(f"Mean IMU dyaw     : {np.mean(imu_dyaw):.6f} rad")
    print("-" * 80)
    print(f"Correlation (GT vs Model) : {np.corrcoef(gt_dyaw, model_dyaw)[0,1]:.4f}")
    print(f"Correlation (GT vs IMU)   : {np.corrcoef(gt_dyaw, imu_dyaw)[0,1]:.4f}")
    print("=" * 80)

    # 6. Plot the comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax1 = axes[0]
    ax1.plot(frames, gt_dyaw, 'b-', label='Ground Truth (GT)', linewidth=2)
    ax1.plot(frames, imu_dyaw, 'r--', label='IMU derived (ωz * dt)', linewidth=2)
    ax1.plot(frames, model_dyaw, 'g:', label='Model Prediction', linewidth=2)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('dyaw (rad)')
    ax1.set_title('Yaw Change Per Frame (Frames 200-250)')
    ax1.legend()
    ax1.grid(True)

    ax2 = axes[1]
    # Cumulative yaw (heading)
    cum_gt = np.cumsum(gt_dyaw)
    cum_imu = np.cumsum(imu_dyaw)
    cum_model = np.cumsum(model_dyaw)
    ax2.plot(frames, cum_gt, 'b-', label='GT Cumulative Yaw', linewidth=2)
    ax2.plot(frames, cum_imu, 'r--', label='IMU Cumulative Yaw', linewidth=2)
    ax2.plot(frames, cum_model, 'g:', label='Model Cumulative Yaw', linewidth=2)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Cumulative Yaw (rad)')
    ax2.set_title('Cumulative Heading')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    out_dir = PROJECT_ROOT / "runs" / "eval" / "gnss_denied"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "yaw_verification.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to: {out_path}")
    print(f"Open it with: open {out_path}")


if __name__ == "__main__":
    main()