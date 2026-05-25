import numpy as np
import torch
from src.config import Config
from src.kitti_loader import load_kitti_sequence
from src.train.train_fusion import train_fusion
from src.eval.metrics import ATE, RPE, heading_error

if __name__ == "__main__":
    config = Config()
    print("Loading and preprocessing KITTI odometry...")
    try:
        bevs, imus, gps, cams, targets = load_kitti_sequence(config)
        print(f"Loaded {len(bevs)} frames.")
    except Exception as e:
        print("Error loading KITTI data:", e)
        print("Using synthetic data generator instead.")
        # Fallback to synthetic data (optional)
        from synthetic_data import generate_sequence
        data = generate_sequence(kind="urban", dur=120, seed=42)
        # Convert to needed format (simplified)
        bevs = [np.zeros((100,100))] * len(data['timestamps'])
        imus = np.column_stack([data['imu_acc'], data['imu_gyro']])
        gps = data['gnss_pos']
        cams = [np.zeros((128,416,3))] * len(data['timestamps'])
        targets = np.column_stack([data['gt_pos'][:,:2], np.linalg.norm(data['gt_vel'], axis=1), data['gt_att'][:,2]])

    print("Training fusion model...")
    model, predictions = train_fusion(config, bevs, imus, gps, cams, targets)

    # If predictions are available, evaluate
    if predictions is not None:
        # Align predictions with ground truth
        gt_pos = targets[:, :2]
        ate = ATE(predictions[:,:2], gt_pos)
        rpe = RPE(predictions[:,:2], gt_pos)
        print(f"ATE: {ate:.3f} m,  RPE: {rpe:.3f} m")