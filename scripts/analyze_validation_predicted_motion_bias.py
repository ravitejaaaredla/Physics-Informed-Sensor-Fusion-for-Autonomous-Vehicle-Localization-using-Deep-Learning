from collections import defaultdict
import csv
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.kitti_frame_pairs import KittiFramePairDataset
from src.models.sensor_fusion import SensorFusionMotionModel
from scripts.train import ROOT, IMUNormalizer, prepare_batch


CHECKPOINT = (
    ROOT
    / "runs"
    / "expanded_train"
    / "best_model.pth"
)

OUTPUT = (
    ROOT
    / "results"
    / "validation_predicted_motion_bias.csv"
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

use_amp = device.type == "cuda"


checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False,
)

stats = checkpoint[
    "training_statistics"
]


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


model = SensorFusionMotionModel().to(
    device
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()


normalizer = IMUNormalizer(
    stats,
    device,
)


predictions = []
yaw_rates = []


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

        yaw_rate = (
            0.5
            * (
                wz_t
                + wz_t1
            )
        )


        predictions.extend(
            pred.float()
            .cpu()
            .numpy()
            .tolist()
        )

        yaw_rates.extend(
            yaw_rate.float()
            .cpu()
            .numpy()
            .tolist()
        )


def speed_regime(step):

    if step < 0.05:
        return "<0.05"

    if step < 0.20:
        return "0.05-0.20"

    if step < 0.50:
        return "0.20-0.50"

    if step < 1.00:
        return "0.50-1.00"

    return ">=1.00"


def turn_regime(wz):

    a = abs(wz)

    if a < 0.02:
        return "straight_<0.02"

    if a < 0.10:
        return "mild_0.02-0.10"

    return "turn_>=0.10"


records = []

for i, row in enumerate(
    dataset.rows
):

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


    pred_step = math.hypot(
        pred_dx,
        pred_dy,
    )

    gt_step = math.hypot(
        gt_dx,
        gt_dy,
    )


    records.append({
        "drive":
            row["drive"],

        "pred_step":
            pred_step,

        "gt_step":
            gt_step,

        "dx_error":
            pred_dx - gt_dx,

        "dy_error":
            pred_dy - gt_dy,

        "step_error":
            pred_step - gt_step,

        "yaw_rate":
            float(
                yaw_rates[i]
            ),

        "speed_regime":
            speed_regime(
                pred_step
            ),

        "turn_regime":
            turn_regime(
                yaw_rates[i]
            ),
    })


speed_order = [
    "<0.05",
    "0.05-0.20",
    "0.20-0.50",
    "0.50-1.00",
    ">=1.00",
]


print()
print("=" * 120)
print("FROZEN MODEL — BIAS BY PREDICTED MOTION REGIME")
print("=" * 120)


summary_rows = []


for drive in sorted({
    r["drive"]
    for r in records
}):

    print()
    print(drive)

    print(
        f"{'Predicted regime':<18}"
        f"{'N':>7}"
        f"{'PredMean':>11}"
        f"{'GTmean':>11}"
        f"{'dx bias cm':>12}"
        f"{'dy bias cm':>12}"
        f"{'step bias cm':>14}"
    )

    print("-" * 85)


    for regime in speed_order:

        group = [
            r
            for r in records
            if (
                r["drive"] == drive
                and
                r["speed_regime"] == regime
            )
        ]

        if not group:
            continue


        dx_bias = float(
            np.mean([
                r["dx_error"]
                for r in group
            ])
        )

        dy_bias = float(
            np.mean([
                r["dy_error"]
                for r in group
            ])
        )

        step_bias = float(
            np.mean([
                r["step_error"]
                for r in group
            ])
        )


        pred_mean = float(
            np.mean([
                r["pred_step"]
                for r in group
            ])
        )

        gt_mean = float(
            np.mean([
                r["gt_step"]
                for r in group
            ])
        )


        print(
            f"{regime:<18}"
            f"{len(group):>7d}"
            f"{pred_mean:>11.4f}"
            f"{gt_mean:>11.4f}"
            f"{100*dx_bias:>12.3f}"
            f"{100*dy_bias:>12.3f}"
            f"{100*step_bias:>14.3f}"
        )


        summary_rows.append({
            "drive":
                drive,

            "predicted_motion_regime":
                regime,

            "n":
                len(group),

            "pred_step_mean_m":
                pred_mean,

            "gt_step_mean_m":
                gt_mean,

            "dx_bias_m":
                dx_bias,

            "dy_bias_m":
                dy_bias,

            "step_bias_m":
                step_bias,
        })


print()
print("=" * 120)
print("FROZEN MODEL — BIAS BY IMU TURN RATE")
print("=" * 120)


turn_order = [
    "straight_<0.02",
    "mild_0.02-0.10",
    "turn_>=0.10",
]


for drive in sorted({
    r["drive"]
    for r in records
}):

    print()
    print(drive)

    print(
        f"{'Turn regime':<20}"
        f"{'N':>7}"
        f"{'|wz| mean':>12}"
        f"{'dx bias cm':>12}"
        f"{'dy bias cm':>12}"
        f"{'step bias cm':>14}"
    )

    print("-" * 80)


    for regime in turn_order:

        group = [
            r
            for r in records
            if (
                r["drive"] == drive
                and
                r["turn_regime"] == regime
            )
        ]

        if not group:
            continue


        print(
            f"{regime:<20}"
            f"{len(group):>7d}"
            f"{np.mean([abs(r['yaw_rate']) for r in group]):>12.4f}"
            f"{100*np.mean([r['dx_error'] for r in group]):>12.3f}"
            f"{100*np.mean([r['dy_error'] for r in group]):>12.3f}"
            f"{100*np.mean([r['step_error'] for r in group]):>14.3f}"
        )


with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        summary_rows
    )


print()
print("FINAL TEST LOADED: False")
print("DRIVE 0018 LOADED: False")
print()
print("Saved:", OUTPUT)
print()
print(
    "PASS: frozen-model conditional bias analysis "
    "completed on validation only."
)
print("=" * 120)
