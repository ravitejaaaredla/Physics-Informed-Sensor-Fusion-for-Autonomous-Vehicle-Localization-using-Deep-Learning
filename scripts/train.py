from pathlib import Path
import argparse
import csv
import json
import math
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path.cwd()

sys.path.insert(
    0,
    str(ROOT),
)

from src.data.kitti_frame_pairs import (
    KittiFramePairDataset,
)

from src.models.sensor_fusion import (
    SensorFusionMotionModel,
)


EARTH_RADIUS_M = 6378137.0


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Physics
# ============================================================

def wrap_angle(angle):

    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


def vehicle_to_east_north(
    dx,
    dy,
    yaw,
):

    c = math.cos(yaw)
    s = math.sin(yaw)

    east = (
        c * dx
        - s * dy
    )

    north = (
        s * dx
        + c * dy
    )

    return east, north


def propagate_latlon(
    lat_deg,
    lon_deg,
    east_m,
    north_m,
):

    lat_rad = math.radians(
        lat_deg
    )

    next_lat_rad = (
        lat_rad
        + north_m / EARTH_RADIUS_M
    )

    next_lat_deg = math.degrees(
        next_lat_rad
    )

    mean_lat_rad = 0.5 * (
        lat_rad
        + next_lat_rad
    )

    next_lon_deg = (
        lon_deg
        + math.degrees(
            east_m
            / (
                EARTH_RADIUS_M
                * math.cos(
                    mean_lat_rad
                )
            )
        )
    )

    return (
        next_lat_deg,
        next_lon_deg,
    )


def latlon_error_m(
    gt_lat,
    gt_lon,
    pred_lat,
    pred_lon,
):

    mean_lat = math.radians(
        0.5 * (
            gt_lat
            + pred_lat
        )
    )

    north = (
        EARTH_RADIUS_M
        * math.radians(
            pred_lat
            - gt_lat
        )
    )

    east = (
        EARTH_RADIUS_M
        * math.cos(mean_lat)
        * math.radians(
            pred_lon
            - gt_lon
        )
    )

    return math.hypot(
        east,
        north,
    )


# ============================================================
# Training-only IMU normalization
# ============================================================

class IMUNormalizer:

    def __init__(
        self,
        stats,
        device,
    ):

        self.mean = torch.tensor(
            stats["imu_mean"],
            dtype=torch.float32,
            device=device,
        ).view(
            1,
            1,
            6,
        )

        self.std = torch.tensor(
            stats["imu_std"],
            dtype=torch.float32,
            device=device,
        ).view(
            1,
            1,
            6,
        )


    def __call__(self, imu):

        return (
            imu - self.mean
        ) / self.std


# ============================================================
# Motion balancing
# ============================================================

def motion_weights(
    target,
    stats,
):

    step = torch.linalg.vector_norm(
        target,
        dim=1,
    )

    result = torch.empty_like(
        step
    )

    w = stats[
        "motion_regime_weights"
    ]

    result[
        step < 0.02
    ] = float(
        w["<0.02"]
    )

    mask = (
        (step >= 0.02)
        & (step < 0.20)
    )

    result[mask] = float(
        w["0.02-0.20"]
    )

    mask = (
        (step >= 0.20)
        & (step < 0.50)
    )

    result[mask] = float(
        w["0.20-0.50"]
    )

    mask = (
        (step >= 0.50)
        & (step < 1.00)
    )

    result[mask] = float(
        w["0.50-1.00"]
    )

    result[
        step >= 1.00
    ] = float(
        w[">=1.00"]
    )

    return result


# ============================================================
# Final training objective
# ============================================================

def compute_loss(
    prediction,
    target,
    stats,
):

    gt_step = torch.linalg.vector_norm(
        target,
        dim=1,
    )

    pred_step = torch.linalg.vector_norm(
        prediction,
        dim=1,
    )

    weights = motion_weights(
        target,
        stats,
    )


    # --------------------------------------------------------
    # Local vehicle-frame dx/dy supervision
    # --------------------------------------------------------

    vector_per_sample = (
        F.smooth_l1_loss(
            prediction,
            target,
            reduction="none",
        )
        .mean(dim=1)
    )

    vector_loss = (
        vector_per_sample
        * weights
    ).mean()


    # --------------------------------------------------------
    # Translation magnitude supervision
    # --------------------------------------------------------

    step_per_sample = (
        F.smooth_l1_loss(
            pred_step,
            gt_step,
            reduction="none",
        )
    )

    step_loss = (
        step_per_sample
        * weights
    ).mean()


    # --------------------------------------------------------
    # Near-stationary constraint
    #
    # Applied using TRAINING TARGETS ONLY.
    # --------------------------------------------------------

    stationary_mask = (
        gt_step < 0.02
    )

    stationary_penalty = (
        pred_step.pow(2)
        * stationary_mask.float()
    )

    stationary_loss = (
        stationary_penalty.sum()
        / stationary_mask.float()
        .sum()
        .clamp_min(1.0)
    )


    total_loss = (
        vector_loss
        + 0.5 * step_loss
        + 1.0 * stationary_loss
    )


    return {
        "total": total_loss,
        "vector": vector_loss,
        "step": step_loss,
        "stationary": stationary_loss,
    }


# ============================================================
# Batch preparation
# ============================================================

def prepare_batch(
    batch,
    device,
    normalizer,
):

    camera_t = batch[
        "camera_t"
    ].to(
        device,
        non_blocking=True,
    )

    camera_t1 = batch[
        "camera_t1"
    ].to(
        device,
        non_blocking=True,
    )

    lidar_t = batch[
        "lidar_t"
    ].to(
        device,
        non_blocking=True,
    )

    lidar_t1 = batch[
        "lidar_t1"
    ].to(
        device,
        non_blocking=True,
    )

    imu_raw = batch[
        "imu_window"
    ].to(
        device,
        non_blocking=True,
    )

    imu = normalizer(
        imu_raw
    )

    target = batch[
        "target_dxdy"
    ].to(
        device,
        non_blocking=True,
    )

    dt = batch[
        "dt"
    ].to(
        device,
        non_blocking=True,
    )

    return (
        camera_t,
        camera_t1,
        lidar_t,
        lidar_t1,
        imu_raw,
        imu,
        target,
        dt,
    )


# ============================================================
# Training epoch
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    normalizer,
    stats,
    use_amp,
):

    model.train()

    totals = {
        "total": 0.0,
        "vector": 0.0,
        "step": 0.0,
        "stationary": 0.0,
    }

    n_samples = 0

    squared_translation_error = 0.0


    for batch in loader:

        (
            camera_t,
            camera_t1,
            lidar_t,
            lidar_t1,
            imu_raw,
            imu,
            target,
            dt,
        ) = prepare_batch(
            batch,
            device,
            normalizer,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):

            prediction = model(
                camera_t,
                camera_t1,
                lidar_t,
                lidar_t1,
                imu,
            )

            losses = compute_loss(
                prediction,
                target,
                stats,
            )


        scaler.scale(
            losses["total"]
        ).backward()


        scaler.unscale_(
            optimizer
        )


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )


        scaler.step(
            optimizer
        )

        scaler.update()


        n = target.shape[0]

        n_samples += n


        for key in totals:

            totals[key] += (
                float(
                    losses[key]
                    .detach()
                )
                * n
            )


        error = (
            prediction.detach()
            - target
        )

        squared_translation_error += float(
            error.pow(2)
            .sum(dim=1)
            .sum()
            .cpu()
        )


    return {
        "total_loss":
            totals["total"]
            / n_samples,

        "vector_loss":
            totals["vector"]
            / n_samples,

        "step_loss":
            totals["step"]
            / n_samples,

        "stationary_loss":
            totals["stationary"]
            / n_samples,

        "translation_rmse_m":
            math.sqrt(
                squared_translation_error
                / n_samples
            ),
    }


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    dataset,
    device,
    normalizer,
    stats,
    use_amp,
):

    model.eval()

    total_samples = 0
    total_loss = 0.0

    squared_translation_error = 0.0
    squared_step_error = 0.0

    predictions = []
    imu_dyaw = []


    for batch in loader:

        (
            camera_t,
            camera_t1,
            lidar_t,
            lidar_t1,
            imu_raw,
            imu,
            target,
            dt,
        ) = prepare_batch(
            batch,
            device,
            normalizer,
        )


        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):

            prediction = model(
                camera_t,
                camera_t1,
                lidar_t,
                lidar_t1,
                imu,
            )

            losses = compute_loss(
                prediction,
                target,
                stats,
            )


        n = target.shape[0]

        total_samples += n

        total_loss += (
            float(
                losses["total"]
            )
            * n
        )


        error = (
            prediction
            - target
        )

        squared_translation_error += float(
            error.pow(2)
            .sum(dim=1)
            .sum()
            .cpu()
        )


        gt_step = torch.linalg.vector_norm(
            target,
            dim=1,
        )

        pred_step = torch.linalg.vector_norm(
            prediction,
            dim=1,
        )

        squared_step_error += float(
            (
                pred_step
                - gt_step
            ).pow(2)
            .sum()
            .cpu()
        )


        # IMU heading increment from raw wz.
        wz_t = imu_raw[
            :,
            -2,
            5,
        ]

        wz_t1 = imu_raw[
            :,
            -1,
            5,
        ]

        batch_dyaw = (
            0.5
            * (
                wz_t + wz_t1
            )
            * dt
        )


        predictions.extend(
            prediction
            .float()
            .cpu()
            .numpy()
            .tolist()
        )

        imu_dyaw.extend(
            batch_dyaw
            .float()
            .cpu()
            .numpy()
            .tolist()
        )


    # --------------------------------------------------------
    # Recursive global-position validation
    # --------------------------------------------------------

    position_errors = []

    previous_key = None
    previous_frame_t1 = None

    state_lat = None
    state_lon = None
    state_yaw = None


    for i, row in enumerate(
        dataset.rows
    ):

        drive = row["drive"]

        segment = int(
            row["segment_id"]
        )

        frame_t = int(
            row["frame_t"]
        )

        frame_t1 = int(
            row["frame_t1"]
        )


        key = (
            drive,
            segment,
        )


        new_sequence = (
            key != previous_key
            or previous_frame_t1 is None
            or frame_t != previous_frame_t1
        )


        if new_sequence:

            # Global anchor exactly once
            # at start of this validation sequence.

            state_lat = float(
                row["gt_lat_t"]
            )

            state_lon = float(
                row["gt_lon_t"]
            )

            state_yaw = float(
                row["gt_yaw_t_rad"]
            )


        pred_dx = float(
            predictions[i][0]
        )

        pred_dy = float(
            predictions[i][1]
        )


        east, north = (
            vehicle_to_east_north(
                pred_dx,
                pred_dy,
                state_yaw,
            )
        )


        (
            next_lat,
            next_lon,
        ) = propagate_latlon(
            state_lat,
            state_lon,
            east,
            north,
        )


        gt_next_lat = float(
            row["gt_lat_t1"]
        )

        gt_next_lon = float(
            row["gt_lon_t1"]
        )


        position_errors.append(
            latlon_error_m(
                gt_next_lat,
                gt_next_lon,
                next_lat,
                next_lon,
            )
        )


        # Recursive predicted state.
        # No GT feedback.
        state_lat = next_lat
        state_lon = next_lon

        state_yaw = wrap_angle(
            state_yaw
            + float(
                imu_dyaw[i]
            )
        )


        previous_key = key
        previous_frame_t1 = (
            frame_t1
        )


    return {
        "loss":
            total_loss
            / total_samples,

        "translation_rmse_m":
            math.sqrt(
                squared_translation_error
                / total_samples
            ),

        "step_rmse_m":
            math.sqrt(
                squared_step_error
                / total_samples
            ),

        "recursive_ate_rmse_m":
            math.sqrt(
                sum(
                    x * x
                    for x in position_errors
                )
                / len(position_errors)
            ),

        "recursive_ate_mean_m":
            float(
                np.mean(
                    position_errors
                )
            ),

        "recursive_ate_median_m":
            float(
                np.median(
                    position_errors
                )
            ),

        "recursive_ate_max_m":
            max(
                position_errors
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
    )

    args = parser.parse_args()


    seed_everything(
        args.seed
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
        device.type == "cuda"
    )


    stats = json.loads(
        (
            ROOT
            / "results"
            / "training_statistics.json"
        ).read_text(
            encoding="utf-8"
        )
    )


    train_dataset = KittiFramePairDataset(
        split="train",
        root=ROOT,
        camera_size=(160, 96),
        bev_size=128,
        imu_window=5,
        cache_lidar=True,
    )


    validation_dataset = (
        KittiFramePairDataset(
            split="validation",
            root=ROOT,
            camera_size=(160, 96),
            bev_size=128,
            imu_window=5,
            cache_lidar=True,
        )
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=use_amp,
    )


    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_amp,
    )


    model = (
        SensorFusionMotionModel()
        .to(device)
    )


    normalizer = IMUNormalizer(
        stats,
        device,
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        )
    )


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )


    run_dir = (
        ROOT
        / "runs"
        / args.run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    history_path = (
        run_dir
        / "history.csv"
    )


    fields = [
        "epoch",
        "lr",

        "train_total_loss",
        "train_vector_loss",
        "train_step_loss",
        "train_stationary_loss",
        "train_translation_rmse_m",

        "val_loss",
        "val_translation_rmse_m",
        "val_step_rmse_m",

        "val_recursive_ate_rmse_m",
        "val_recursive_ate_mean_m",
        "val_recursive_ate_median_m",
        "val_recursive_ate_max_m",

        "epoch_seconds",
    ]


    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        csv.DictWriter(
            f,
            fieldnames=fields,
        ).writeheader()


    best_ate = float(
        "inf"
    )

    best_epoch = None

    stale_epochs = 0


    print()
    print("=" * 96)
    print("SENSOR-FUSION LOCALIZATION TRAINING")
    print("=" * 96)

    print(
        "Device                   :",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU                      :",
            torch.cuda.get_device_name(0),
        )

    print(
        "Train pairs              :",
        len(train_dataset),
    )

    print(
        "Validation pairs         :",
        len(validation_dataset),
    )

    print(
        "Cross-drive 0018 loaded  :",
        False,
    )

    print(
        "Model output             : dx, dy"
    )

    print(
        "Checkpoint selection     : "
        "validation recursive ATE RMSE"
    )

    print(
        "Loss                     : "
        "vector + 0.5*step + stationary"
    )

    print()


    for epoch in range(
        1,
        args.epochs + 1,
    ):

        start = time.time()


        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            normalizer,
            stats,
            use_amp,
        )


        val_metrics = validate(
            model,
            validation_loader,
            validation_dataset,
            device,
            normalizer,
            stats,
            use_amp,
        )


        val_ate = val_metrics[
            "recursive_ate_rmse_m"
        ]


        scheduler.step(
            val_ate
        )


        lr = optimizer.param_groups[
            0
        ]["lr"]


        elapsed = (
            time.time()
            - start
        )


        record = {
            "epoch": epoch,
            "lr": lr,

            "train_total_loss":
                train_metrics[
                    "total_loss"
                ],

            "train_vector_loss":
                train_metrics[
                    "vector_loss"
                ],

            "train_step_loss":
                train_metrics[
                    "step_loss"
                ],

            "train_stationary_loss":
                train_metrics[
                    "stationary_loss"
                ],

            "train_translation_rmse_m":
                train_metrics[
                    "translation_rmse_m"
                ],

            "val_loss":
                val_metrics[
                    "loss"
                ],

            "val_translation_rmse_m":
                val_metrics[
                    "translation_rmse_m"
                ],

            "val_step_rmse_m":
                val_metrics[
                    "step_rmse_m"
                ],

            "val_recursive_ate_rmse_m":
                val_metrics[
                    "recursive_ate_rmse_m"
                ],

            "val_recursive_ate_mean_m":
                val_metrics[
                    "recursive_ate_mean_m"
                ],

            "val_recursive_ate_median_m":
                val_metrics[
                    "recursive_ate_median_m"
                ],

            "val_recursive_ate_max_m":
                val_metrics[
                    "recursive_ate_max_m"
                ],

            "epoch_seconds":
                elapsed,
        }


        with history_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writerow(
                record
            )


        improved = (
            val_ate < best_ate
        )


        if improved:

            best_ate = val_ate
            best_epoch = epoch
            stale_epochs = 0


            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "best_validation_recursive_ate_rmse_m":
                        best_ate,

                    "training_statistics":
                        stats,

                    "model_inputs": [
                        "camera_t",
                        "camera_t1",
                        "lidar_t",
                        "lidar_t1",
                        "imu_window",
                    ],

                    "model_output": [
                        "dx_m",
                        "dy_m",
                    ],

                    "loss": {
                        "vector": 1.0,
                        "step": 0.5,
                        "stationary": 1.0,
                        "stationary_threshold_m":
                            0.02,
                    },

                    "recursive_state": {
                        "initial_position":
                            "GT once at sequence start",

                        "initial_heading":
                            "GT once at sequence start",

                        "future_position":
                            "predicted recursively",

                        "future_heading":
                            "IMU wz integration",
                    },

                    "drive0018_used_for_training":
                        False,

                    "drive0018_used_for_model_selection":
                        False,

                    "args":
                        vars(args),
                },

                run_dir
                / "best_model.pth",
            )


        else:

            stale_epochs += 1


        print(
            f"Epoch {epoch:03d} | "
            f"train total "
            f"{train_metrics['total_loss']:.6f} | "
            f"vector "
            f"{train_metrics['vector_loss']:.6f} | "
            f"step "
            f"{train_metrics['step_loss']:.6f} | "
            f"stationary "
            f"{train_metrics['stationary_loss']:.6f} | "
            f"train RMSE "
            f"{train_metrics['translation_rmse_m']:.4f} m | "
            f"val local RMSE "
            f"{val_metrics['translation_rmse_m']:.4f} m | "
            f"val step RMSE "
            f"{val_metrics['step_rmse_m']:.4f} m | "
            f"val ATE "
            f"{val_ate:.4f} m | "
            f"lr {lr:.2e} | "
            f"{elapsed:.1f}s"
            + (
                " | BEST"
                if improved
                else ""
            )
        )


        if (
            stale_epochs
            >= args.patience
        ):

            print()
            print(
                "Early stopping:",
                args.patience,
                "epochs without validation "
                "ATE improvement."
            )

            break


    print()
    print("=" * 96)
    print("TRAINING COMPLETE")
    print("=" * 96)

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation recursive ATE RMSE:",
        f"{best_ate:.6f} m",
    )

    print(
        "Best checkpoint:",
        run_dir
        / "best_model.pth",
    )

    print(
        "History:",
        history_path,
    )


if __name__ == "__main__":
    main()
