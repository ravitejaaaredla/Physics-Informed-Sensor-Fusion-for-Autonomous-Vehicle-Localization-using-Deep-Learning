from pathlib import Path
from collections import defaultdict
import csv
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.kitti_frame_pairs import KittiFramePairDataset
from src.models.sensor_fusion import SensorFusionMotionModel
from scripts.train import IMUNormalizer, prepare_batch


ROOT = Path.cwd()

CHECKPOINT = (
    ROOT
    / "runs"
    / "expanded_train"
    / "best_model.pth"
)

OUTPUT = (
    ROOT
    / "results"
    / "validation_translation_bias_analysis.csv"
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

use_amp = device.type == "cuda"

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False,
)

stats = checkpoint["training_statistics"]

dataset = KittiFramePairDataset(
    split="validation",
    root=ROOT,
    camera_size=(160, 96),
    bev_size=128,
    imu_window=5,
    cache_lidar=True,
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=use_amp,
)

model = SensorFusionMotionModel().to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

normalizer = IMUNormalizer(
    stats,
    device,
)


predictions = []

with torch.no_grad():

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

            pred = model(
                camera_t,
                camera_t1,
                lidar_t,
                lidar_t1,
                imu,
            )

        predictions.extend(
            pred.float()
            .cpu()
            .numpy()
            .tolist()
        )


if len(predictions) != len(dataset):
    raise RuntimeError(
        "Prediction count mismatch."
    )


metrics = defaultdict(
    lambda: {
        "dx_errors": [],
        "dy_errors": [],
        "step_errors": [],
        "east_errors": [],
        "north_errors": [],
        "gt_distance": 0.0,
    }
)


for i, row in enumerate(dataset.rows):

    drive = row["drive"]

    pred_dx = float(
        predictions[i][0]
    )

    pred_dy = float(
        predictions[i][1]
    )

    gt_dx = float(
        row["gt_dx_m"]
    )

    gt_dy = float(
        row["gt_dy_m"]
    )

    yaw = float(
        row["gt_yaw_t_rad"]
    )


    dx_error = pred_dx - gt_dx
    dy_error = pred_dy - gt_dy

    pred_step = math.hypot(
        pred_dx,
        pred_dy,
    )

    gt_step = math.hypot(
        gt_dx,
        gt_dy,
    )

    step_error = (
        pred_step - gt_step
    )


    # Rotate translation residual into East/North
    # using GT yaw ONLY for offline diagnostic.
    east_error = (
        math.cos(yaw) * dx_error
        -
        math.sin(yaw) * dy_error
    )

    north_error = (
        math.sin(yaw) * dx_error
        +
        math.cos(yaw) * dy_error
    )


    d = metrics[drive]

    d["dx_errors"].append(
        dx_error
    )

    d["dy_errors"].append(
        dy_error
    )

    d["step_errors"].append(
        step_error
    )

    d["east_errors"].append(
        east_error
    )

    d["north_errors"].append(
        north_error
    )

    d["gt_distance"] += gt_step


rows = []


for drive in sorted(metrics):

    d = metrics[drive]

    dx = np.array(
        d["dx_errors"]
    )

    dy = np.array(
        d["dy_errors"]
    )

    step = np.array(
        d["step_errors"]
    )

    east = np.array(
        d["east_errors"]
    )

    north = np.array(
        d["north_errors"]
    )


    cumulative_east = float(
        east.sum()
    )

    cumulative_north = float(
        north.sum()
    )

    cumulative_drift = math.hypot(
        cumulative_east,
        cumulative_north,
    )


    rows.append({
        "drive":
            drive,

        "pairs":
            len(dx),

        "gt_distance_m":
            d["gt_distance"],

        "mean_dx_bias_m_per_frame":
            float(dx.mean()),

        "mean_dy_bias_m_per_frame":
            float(dy.mean()),

        "mean_step_bias_m_per_frame":
            float(step.mean()),

        "dx_rmse_m":
            float(
                math.sqrt(
                    np.mean(dx ** 2)
                )
            ),

        "dy_rmse_m":
            float(
                math.sqrt(
                    np.mean(dy ** 2)
                )
            ),

        "cumulative_east_error_m":
            cumulative_east,

        "cumulative_north_error_m":
            cumulative_north,

        "translation_only_final_drift_m":
            cumulative_drift,

        "translation_drift_per_100m":
            (
                100.0
                * cumulative_drift
                / d["gt_distance"]
            ),
    })


with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(rows)


print()
print("=" * 132)
print("VALIDATION SIGNED TRANSLATION-BIAS ANALYSIS")
print("=" * 132)

print(
    f"{'Drive':34}"
    f"{'Pairs':>7}"
    f"{'dx bias cm':>12}"
    f"{'dy bias cm':>12}"
    f"{'step bias cm':>14}"
    f"{'dx RMSE':>10}"
    f"{'dy RMSE':>10}"
    f"{'Cum East':>11}"
    f"{'Cum North':>11}"
    f"{'Final drift':>13}"
    f"{'/100m':>9}"
)

print("-" * 132)


for r in rows:

    print(
        f"{r['drive']:<34}"
        f"{r['pairs']:>7d}"
        f"{100*r['mean_dx_bias_m_per_frame']:>12.3f}"
        f"{100*r['mean_dy_bias_m_per_frame']:>12.3f}"
        f"{100*r['mean_step_bias_m_per_frame']:>14.3f}"
        f"{r['dx_rmse_m']:>10.4f}"
        f"{r['dy_rmse_m']:>10.4f}"
        f"{r['cumulative_east_error_m']:>11.3f}"
        f"{r['cumulative_north_error_m']:>11.3f}"
        f"{r['translation_only_final_drift_m']:>13.3f}"
        f"{r['translation_drift_per_100m']:>9.3f}"
    )


print()
print("FINAL TEST LOADED: False")
print("DRIVE 0018 LOADED: False")
print()
print("Saved:", OUTPUT)
print()
print(
    "PASS: signed translation bias measured "
    "on validation only."
)
print("=" * 132)
