import sys
from pathlib import Path
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from src.data.kitti_dataset import KITTIRawDataset
from src.models.fusion_pinn import FusionPINN
from src.physics.physics_loss import compute_total_loss


def train_one_epoch(model, dataloader, optimizer, device, config):
    model.train()
    total_loss = 0.0
    num_batches = 0

    target_mean = torch.tensor(config.target_mean, dtype=torch.float32).to(device)
    target_std = torch.tensor(config.target_std, dtype=torch.float32).to(device)
    lambda_data = config.lambda_data
    lambda_physics = config.lambda_physics
    lambda_imu = config.lambda_imu
    imu_dyaw_scale = config.imu_dyaw_scale

    for batch in dataloader:
        camera = batch["camera"].to(device)
        lidar = batch["lidar_bev"].to(device)
        imu = batch["imu"].to(device)
        target = batch["target"].to(device)

        imu_omega_z = imu[:, 5]  # Extract gyro z

        optimizer.zero_grad()
        pred = model(camera, lidar, imu)

        loss_dict = compute_total_loss(
            pred, target,
            target_mean, target_std,
            imu_omega_z=imu_omega_z,
            lambda_data=lambda_data,
            lambda_physics=lambda_physics,
            lambda_imu=lambda_imu,
            imu_dyaw_scale=imu_dyaw_scale,
            dt=0.1,
        )
        loss = loss_dict["total_loss"]

        # ========== DIAGNOSTIC ==========
        if num_batches % 100 == 0:
            with torch.no_grad():
                imu_feat = model.imu_encoder(imu)
                imu_feat_mean = imu_feat.abs().mean().item()
            print(f"  Batch {num_batches:4d} | Data Loss: {loss_dict['data_loss'].item():.6f} | Physics Loss: {loss_dict['physics_loss'].item():.6f} | IMU Loss: {loss_dict['imu_loss'].item():.6f} | IMU Feat Mean: {imu_feat_mean:.6f}")
        # =================================

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_one_epoch(model, dataloader, device, config):
    if dataloader is None or len(dataloader) == 0:
        return {
            "total_loss": 0.0,
            "data_loss": 0.0,
            "physics_loss": 0.0,
            "imu_loss": 0.0,
        }

    model.eval()
    total_loss_sum = 0.0
    data_loss_sum = 0.0
    physics_loss_sum = 0.0
    imu_loss_sum = 0.0
    num_batches = 0

    target_mean = torch.tensor(config.target_mean, dtype=torch.float32).to(device)
    target_std = torch.tensor(config.target_std, dtype=torch.float32).to(device)
    lambda_data = config.lambda_data
    lambda_physics = config.lambda_physics
    lambda_imu = config.lambda_imu
    imu_dyaw_scale = config.imu_dyaw_scale

    with torch.no_grad():
        for batch in dataloader:
            camera = batch["camera"].to(device)
            lidar = batch["lidar_bev"].to(device)
            imu = batch["imu"].to(device)
            target = batch["target"].to(device)
            imu_omega_z = imu[:, 5]

            pred = model(camera, lidar, imu)
            loss_dict = compute_total_loss(
                pred, target,
                target_mean, target_std,
                imu_omega_z=imu_omega_z,
                lambda_data=lambda_data,
                lambda_physics=lambda_physics,
                lambda_imu=lambda_imu,
                imu_dyaw_scale=imu_dyaw_scale,
                dt=0.1,
            )

            total_loss_sum += loss_dict["total_loss"].item()
            data_loss_sum += loss_dict["data_loss"].item()
            physics_loss_sum += loss_dict["physics_loss"].item()
            imu_loss_sum += loss_dict["imu_loss"].item()
            num_batches += 1

    return {
        "total_loss": total_loss_sum / num_batches,
        "data_loss": data_loss_sum / num_batches,
        "physics_loss": physics_loss_sum / num_batches,
        "imu_loss": imu_loss_sum / num_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Default learning rate for non-IMU layers")
    parser.add_argument("--lr_imu", type=float, default=5e-3, help="Learning rate for IMU branch")
    args = parser.parse_args()

    print("=" * 80)
    print("08 - TRAIN SENSOR FUSION LOCALIZATION MODEL")
    print("=" * 80)
    print("Project root:", PROJECT_ROOT)

    train_dataset = KITTIRawDataset(Config.train_sequence_dirs, max_frames=Config.max_frames)
    val_dataset = KITTIRawDataset(Config.val_sequence_dirs, max_frames=Config.max_frames) if Config.val_sequence_dirs else None

    print("\nTraining drives:")
    for d in Config.train_sequence_dirs:
        print(f"  {d}")

    print("\nValidation drives:")
    if Config.val_sequence_dirs:
        for d in Config.val_sequence_dirs:
            print(f"  {d}")
    else:
        print("  (none)")

    print("\nDataset size")
    print(f"Training samples  : {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset) if val_dataset else 0}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset else None

    model = FusionPINN(Config).to(device)

    # ========== RESUME FROM CHECKPOINT ==========
    checkpoint_path = Config.checkpoint_dir / "model_best.pth"
    if checkpoint_path.exists():
        print(f"\nLoading existing checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("✅ Checkpoint loaded. Continuing training...")
    else:
        print("\nNo checkpoint found. Starting from scratch.")
    # ============================================

    # ========== SEPARATE LEARNING RATES ==========
    imu_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'imu_encoder' in name or 'imu_to_dyaw' in name:
            imu_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.Adam([
        {'params': imu_params, 'lr': args.lr_imu},
        {'params': other_params, 'lr': args.lr}
    ], weight_decay=Config.weight_decay)
    # =============================================

    print("\nTraining setup")
    print(f"Epochs          : {args.epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Default LR      : {args.lr}")
    print(f"IMU branch LR   : {args.lr_imu}")
    print(f"Lambda data     : {Config.lambda_data}")
    print(f"Lambda physics  : {Config.lambda_physics}")
    print(f"Lambda IMU      : {Config.lambda_imu}")
    print(f"IMU dyaw scale  : {Config.imu_dyaw_scale}")

    print("\nStarting training...")

    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, Config)

        if val_loader is not None and len(val_loader) > 0:
            val_metrics = validate_one_epoch(model, val_loader, device, Config)
            val_loss = val_metrics["total_loss"]
            imu_loss_val = val_metrics["imu_loss"]
        else:
            val_loss = None
            imu_loss_val = None

        elapsed = time.time() - start_time

        if val_loss is not None:
            print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | IMU Loss: {imu_loss_val:.6f} | Time: {elapsed:.1f}s")
            current_loss = val_loss
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.6f} | Time: {elapsed:.1f}s")
            current_loss = train_loss

        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            torch.save(model.state_dict(), Config.checkpoint_dir / "model_best.pth")
            print(f"  -> Best model saved (loss: {best_loss:.6f})")

    print("\n" + "=" * 80)
    print(f"Training finished! Best model at epoch {best_epoch} with loss {best_loss:.6f}")
    print(f"Saved to: {Config.checkpoint_dir / 'model_best.pth'}")
    print("=" * 80)


if __name__ == "__main__":
    main()