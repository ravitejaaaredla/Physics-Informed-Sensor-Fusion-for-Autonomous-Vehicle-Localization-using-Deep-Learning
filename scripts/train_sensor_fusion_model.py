import sys
from pathlib import Path
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from src.data.kitti_dataset import KITTIRawDataset
from src.models.fusion_pinn import FusionPINN
from src.physics.physics_loss import compute_total_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()

    total_loss_sum = 0.0
    data_loss_sum = 0.0
    physics_loss_sum = 0.0
    num_batches = 0

    for batch in dataloader:
        camera = batch["camera"].to(device)
        lidar_bev = batch["lidar_bev"].to(device)
        imu = batch["imu"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        predicted_motion = model(camera, lidar_bev, imu)

        losses = compute_total_loss(
            predicted_motion,
            target,
            target_mean=Config.target_mean,
            target_std=Config.target_std,
            lambda_data=Config.lambda_data,
            lambda_physics=Config.lambda_physics,
        )

        loss = losses["total_loss"]
        loss.backward()
        optimizer.step()

        total_loss_sum += float(losses["total_loss"].detach().cpu())
        data_loss_sum += float(losses["data_loss"].detach().cpu())
        physics_loss_sum += float(losses["physics_loss"].detach().cpu())
        num_batches += 1

    return {
        "total_loss": total_loss_sum / num_batches,
        "data_loss": data_loss_sum / num_batches,
        "physics_loss": physics_loss_sum / num_batches,
    }


def validate_one_epoch(model, dataloader, device):
    model.eval()

    total_loss_sum = 0.0
    data_loss_sum = 0.0
    physics_loss_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            camera = batch["camera"].to(device)
            lidar_bev = batch["lidar_bev"].to(device)
            imu = batch["imu"].to(device)
            target = batch["target"].to(device)

            predicted_motion = model(camera, lidar_bev, imu)

            losses = compute_total_loss(
                predicted_motion,
                target,
                target_mean=Config.target_mean,
                target_std=Config.target_std,
                lambda_data=Config.lambda_data,
                lambda_physics=Config.lambda_physics,
            )

            total_loss_sum += float(losses["total_loss"].detach().cpu())
            data_loss_sum += float(losses["data_loss"].detach().cpu())
            physics_loss_sum += float(losses["physics_loss"].detach().cpu())
            num_batches += 1

    return {
        "total_loss": total_loss_sum / num_batches,
        "data_loss": data_loss_sum / num_batches,
        "physics_loss": physics_loss_sum / num_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    print("=" * 80)
    print("08 - TRAIN SENSOR FUSION LOCALIZATION MODEL")
    print("=" * 80)

    set_seed(Config.seed)

    print("Project root:", PROJECT_ROOT)

    print("\nTraining drives:")
    for p in Config.train_sequence_dirs:
        print(" ", p)

    print("\nValidation drives:")
    for p in Config.val_sequence_dirs:
        print(" ", p)

    train_dataset = KITTIRawDataset(
        Config.train_sequence_dirs,
        max_frames=Config.max_frames,
    )

    val_dataset = KITTIRawDataset(
        Config.val_sequence_dirs,
        max_frames=Config.max_frames,
    )

    print("\nDataset size")
    print("Training samples  :", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)

    model = FusionPINN(Config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
    )

    Config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_model_path = Config.checkpoint_dir / "model_best.pth"
    last_model_path = Config.checkpoint_dir / "model_last.pth"

    print("\nTraining setup")
    print("Epochs       :", args.epochs)
    print("Batch size   :", Config.batch_size)
    print("Learning rate:", Config.learning_rate)

    print("\nStarting training...")

    for epoch in range(1, args.epochs + 1):
        train_losses = train_one_epoch(model, train_loader, optimizer, device)
        val_losses = validate_one_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Total: {train_losses['total_loss']:.8f} | "
            f"Train Data: {train_losses['data_loss']:.8f} | "
            f"Train Physics: {train_losses['physics_loss']:.8f} | "
            f"Val Total: {val_losses['total_loss']:.8f} | "
            f"Val Data: {val_losses['data_loss']:.8f} | "
            f"Val Physics: {val_losses['physics_loss']:.8f}"
        )

        torch.save(model.state_dict(), last_model_path)

        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            torch.save(model.state_dict(), best_model_path)
            print("  Saved best model:", best_model_path)

    print("\nDONE")
    print("Best validation loss:", best_val_loss)
    print("Best model saved at:")
    print(best_model_path)


if __name__ == "__main__":
    main()
