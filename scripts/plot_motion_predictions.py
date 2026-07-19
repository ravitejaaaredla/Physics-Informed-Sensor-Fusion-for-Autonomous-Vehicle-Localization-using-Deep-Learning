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


def plot_motion_predictions():
    # Create output directory
    output_dir = Config.eval_dir / "gnss_denied"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (test drive)
    dataset = KITTIRawDataset([Config.test_sequence_dir], max_frames=Config.max_frames)
    print(f"Loaded {len(dataset)} samples from test drive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionPINN(Config).to(device)
    model_path = Config.checkpoint_dir / "model_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model: {model_path}")

    target_mean = np.array(Config.target_mean)
    target_std = np.array(Config.target_std)

    # We'll analyze frames 200-260 (where denial happens)
    start_frame = 200
    end_frame = min(260, len(dataset))

    preds = []
    targets = []
    frames = []

    for idx in range(start_frame, end_frame):
        sample = dataset[idx]
        camera = sample["camera"].unsqueeze(0).to(device)
        lidar = sample["lidar_bev"].unsqueeze(0).to(device)
        imu = sample["imu"].unsqueeze(0).to(device)
        target = sample["target"].numpy()  # normalized target

        with torch.no_grad():
            pred_norm = model(camera, lidar, imu).squeeze().cpu().numpy()

        # Denormalize both
        pred_phys = (pred_norm * target_std) + target_mean
        target_phys = (target * target_std) + target_mean

        preds.append(pred_phys)
        targets.append(target_phys)
        frames.append(idx)

    preds = np.array(preds)
    targets = np.array(targets)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    titles = ['dx_body (Forward displacement)', 'dy_body (Lateral displacement)',
              'dv (Velocity change)', 'dyaw (Yaw change)']
    units = ['[m]', '[m]', '[m/s]', '[rad]']

    for i, ax in enumerate(axes.flat):
        ax.plot(frames, targets[:, i], 'b-', label='Ground Truth', linewidth=2)
        ax.plot(frames, preds[:, i], 'r--', label='PINN Prediction', linewidth=2)
        ax.set_xlabel('Frame')
        ax.set_ylabel(f'{titles[i]} {units[i]}')
        ax.set_title(titles[i])
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    out_path = output_dir / "motion_analysis.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved motion analysis plot to: {out_path}")

    # Calculate and print average errors
    errors = np.abs(preds - targets)
    print("\nAverage absolute errors (frames 200-259):")
    print(f"  dx_body: {np.mean(errors[:, 0]):.4f} m")
    print(f"  dy_body: {np.mean(errors[:, 1]):.6f} m")
    print(f"  dv:      {np.mean(errors[:, 2]):.4f} m/s")
    print(f"  dyaw:    {np.mean(errors[:, 3]):.6f} rad ({np.degrees(np.mean(errors[:, 3])):.4f} deg)")


if __name__ == "__main__":
    plot_motion_predictions()