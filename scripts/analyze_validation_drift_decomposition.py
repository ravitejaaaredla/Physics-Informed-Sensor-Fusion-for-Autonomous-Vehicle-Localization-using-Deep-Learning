from pathlib import Path
from collections import defaultdict
import csv
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.kitti_frame_pairs import KittiFramePairDataset
from src.models.sensor_fusion import SensorFusionMotionModel

from scripts.train import (
    IMUNormalizer,
    vehicle_to_east_north,
    propagate_latlon,
    latlon_error_m,
    wrap_angle,
)


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
    / "validation_drift_decomposition.csv"
)


# ============================================================
# Load frozen selected checkpoint
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

use_amp = (
    device.type == "cuda"
)

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
    checkpoint["model_state_dict"]
)

model.eval()


normalizer = IMUNormalizer(
    stats,
    device,
)


# ============================================================
# Neural translation + raw-IMU yaw increments
# ============================================================

predictions = []
imu_dyaw = []


with torch.no_grad():

    for batch in loader:

        camera_t = batch[
            "camera_t"
        ].to(device)

        camera_t1 = batch[
            "camera_t1"
        ].to(device)

        lidar_t = batch[
            "lidar_t"
        ].to(device)

        lidar_t1 = batch[
            "lidar_t1"
        ].to(device)

        imu_raw = batch[
            "imu_window"
        ].to(device)

        imu = normalizer(
            imu_raw
        )

        dt = batch[
            "dt"
        ].to(device)

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

        # Same physics used by the actual system.
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

        dyaw = (
            0.5
            * (
                wz_t
                + wz_t1
            )
            * dt
        )

        predictions.extend(
            pred.float()
            .cpu()
            .numpy()
            .tolist()
        )

        imu_dyaw.extend(
            dyaw.float()
            .cpu()
            .numpy()
            .tolist()
        )


if len(predictions) != len(dataset):
    raise RuntimeError(
        "Prediction count mismatch."
    )


# ============================================================
# Metrics storage
# ============================================================

metrics = defaultdict(
    lambda: {
        "pairs": 0,
        "gt_distance": 0.0,

        "A_errors": [],
        "B_errors": [],
        "C_errors": [],
        "D_errors": [],

        "yaw_abs_errors_deg": [],
        "final_yaw_error_deg": 0.0,
    }
)


# ============================================================
# Recursive states
# ============================================================

previous_key = None
previous_frame_t1 = None

# A = predicted translation + IMU yaw
A_lat = A_lon = A_yaw = None

# B = GT translation + IMU yaw
B_lat = B_lon = B_yaw = None

# C = predicted translation + GT yaw
C_lat = C_lon = None

# D = GT translation + GT yaw
D_lat = D_lon = None


for i, row in enumerate(
    dataset.rows
):

    drive = row[
        "drive"
    ]

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


    gt_lat_t = float(
        row["gt_lat_t"]
    )

    gt_lon_t = float(
        row["gt_lon_t"]
    )

    gt_lat_t1 = float(
        row["gt_lat_t1"]
    )

    gt_lon_t1 = float(
        row["gt_lon_t1"]
    )

    gt_yaw_t = float(
        row["gt_yaw_t_rad"]
    )

    gt_yaw_t1 = float(
        row["gt_yaw_t1_rad"]
    )


    if new_sequence:

        A_lat = gt_lat_t
        A_lon = gt_lon_t
        A_yaw = gt_yaw_t

        B_lat = gt_lat_t
        B_lon = gt_lon_t
        B_yaw = gt_yaw_t

        C_lat = gt_lat_t
        C_lon = gt_lon_t

        D_lat = gt_lat_t
        D_lon = gt_lon_t


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

    gt_step = math.hypot(
        gt_dx,
        gt_dy,
    )

    dyaw_imu = float(
        imu_dyaw[i]
    )


    # --------------------------------------------------------
    # A. ACTUAL SYSTEM
    # predicted dx/dy + recursively integrated IMU yaw
    # --------------------------------------------------------

    east, north = vehicle_to_east_north(
        pred_dx,
        pred_dy,
        A_yaw,
    )

    A_next_lat, A_next_lon = propagate_latlon(
        A_lat,
        A_lon,
        east,
        north,
    )

    A_error = latlon_error_m(
        gt_lat_t1,
        gt_lon_t1,
        A_next_lat,
        A_next_lon,
    )


    # --------------------------------------------------------
    # B. YAW/IMU DRIFT ISOLATION
    # GT dx/dy + recursively integrated IMU yaw
    # --------------------------------------------------------

    east, north = vehicle_to_east_north(
        gt_dx,
        gt_dy,
        B_yaw,
    )

    B_next_lat, B_next_lon = propagate_latlon(
        B_lat,
        B_lon,
        east,
        north,
    )

    B_error = latlon_error_m(
        gt_lat_t1,
        gt_lon_t1,
        B_next_lat,
        B_next_lon,
    )


    # --------------------------------------------------------
    # C. TRANSLATION DRIFT ISOLATION
    # predicted dx/dy + exact GT yaw for this frame
    # --------------------------------------------------------

    east, north = vehicle_to_east_north(
        pred_dx,
        pred_dy,
        gt_yaw_t,
    )

    C_next_lat, C_next_lon = propagate_latlon(
        C_lat,
        C_lon,
        east,
        north,
    )

    C_error = latlon_error_m(
        gt_lat_t1,
        gt_lon_t1,
        C_next_lat,
        C_next_lon,
    )


    # --------------------------------------------------------
    # D. ORACLE SANITY CHECK
    # GT dx/dy + GT yaw
    # should reconstruct GT trajectory almost exactly
    # --------------------------------------------------------

    east, north = vehicle_to_east_north(
        gt_dx,
        gt_dy,
        gt_yaw_t,
    )

    D_next_lat, D_next_lon = propagate_latlon(
        D_lat,
        D_lon,
        east,
        north,
    )

    D_error = latlon_error_m(
        gt_lat_t1,
        gt_lon_t1,
        D_next_lat,
        D_next_lon,
    )


    # --------------------------------------------------------
    # IMU yaw error
    # --------------------------------------------------------

    A_next_yaw = wrap_angle(
        A_yaw
        + dyaw_imu
    )

    B_next_yaw = wrap_angle(
        B_yaw
        + dyaw_imu
    )

    yaw_error_rad = wrap_angle(
        A_next_yaw
        - gt_yaw_t1
    )

    yaw_error_deg = math.degrees(
        yaw_error_rad
    )


    d = metrics[
        drive
    ]

    d["pairs"] += 1

    d["gt_distance"] += (
        gt_step
    )

    d["A_errors"].append(
        A_error
    )

    d["B_errors"].append(
        B_error
    )

    d["C_errors"].append(
        C_error
    )

    d["D_errors"].append(
        D_error
    )

    d["yaw_abs_errors_deg"].append(
        abs(
            yaw_error_deg
        )
    )

    d["final_yaw_error_deg"] = (
        yaw_error_deg
    )


    # --------------------------------------------------------
    # Recursive updates
    # --------------------------------------------------------

    A_lat = A_next_lat
    A_lon = A_next_lon
    A_yaw = A_next_yaw

    B_lat = B_next_lat
    B_lon = B_next_lon
    B_yaw = B_next_yaw

    C_lat = C_next_lat
    C_lon = C_next_lon

    D_lat = D_next_lat
    D_lon = D_next_lon


    previous_key = key
    previous_frame_t1 = (
        frame_t1
    )


# ============================================================
# Summaries
# ============================================================

def rmse(values):

    return math.sqrt(
        sum(
            x * x
            for x in values
        )
        / len(values)
    )


summary_rows = []


for drive in sorted(
    metrics
):

    d = metrics[
        drive
    ]

    distance = d[
        "gt_distance"
    ]

    A_final = d[
        "A_errors"
    ][-1]

    B_final = d[
        "B_errors"
    ][-1]

    C_final = d[
        "C_errors"
    ][-1]

    D_final = d[
        "D_errors"
    ][-1]


    summary_rows.append({

        "drive":
            drive,

        "pairs":
            d["pairs"],

        "gt_distance_m":
            distance,

        "full_system_ate_rmse_m":
            rmse(
                d["A_errors"]
            ),

        "full_system_final_error_m":
            A_final,

        "full_system_final_drift_per_100m":
            100.0
            * A_final
            / distance,

        "imu_yaw_only_ate_rmse_m":
            rmse(
                d["B_errors"]
            ),

        "imu_yaw_only_final_error_m":
            B_final,

        "imu_yaw_only_final_drift_per_100m":
            100.0
            * B_final
            / distance,

        "translation_only_ate_rmse_m":
            rmse(
                d["C_errors"]
            ),

        "translation_only_final_error_m":
            C_final,

        "translation_only_final_drift_per_100m":
            100.0
            * C_final
            / distance,

        "oracle_ate_rmse_m":
            rmse(
                d["D_errors"]
            ),

        "oracle_final_error_m":
            D_final,

        "imu_yaw_mae_deg":
            float(
                np.mean(
                    d[
                        "yaw_abs_errors_deg"
                    ]
                )
            ),

        "imu_yaw_final_error_deg":
            d[
                "final_yaw_error_deg"
            ],
    })


# ============================================================
# Save
# ============================================================

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


# ============================================================
# Console report
# ============================================================

print()
print("=" * 142)
print("VALIDATION RECURSIVE DRIFT DECOMPOSITION")
print("=" * 142)

print()
print(
    "A = predicted dx/dy + IMU yaw   (actual system)"
)

print(
    "B = GT dx/dy + IMU yaw          (heading/IMU drift)"
)

print(
    "C = predicted dx/dy + GT yaw    (translation-network drift)"
)

print(
    "D = GT dx/dy + GT yaw           (oracle sanity check)"
)

print()

print(
    f"{'Drive':34}"
    f"{'GTdist':>9}"
    f"{'A_ATE':>9}"
    f"{'A_Final':>10}"
    f"{'A/100m':>9}"
    f"{'B_ATE':>9}"
    f"{'B_Final':>10}"
    f"{'C_ATE':>9}"
    f"{'C_Final':>10}"
    f"{'YawMAE°':>9}"
    f"{'YawFinal°':>11}"
    f"{'Oracle':>9}"
)

print("-" * 142)


for r in summary_rows:

    print(
        f"{r['drive']:<34}"
        f"{r['gt_distance_m']:>9.1f}"
        f"{r['full_system_ate_rmse_m']:>9.3f}"
        f"{r['full_system_final_error_m']:>10.3f}"
        f"{r['full_system_final_drift_per_100m']:>9.3f}"
        f"{r['imu_yaw_only_ate_rmse_m']:>9.3f}"
        f"{r['imu_yaw_only_final_error_m']:>10.3f}"
        f"{r['translation_only_ate_rmse_m']:>9.3f}"
        f"{r['translation_only_final_error_m']:>10.3f}"
        f"{r['imu_yaw_mae_deg']:>9.3f}"
        f"{r['imu_yaw_final_error_deg']:>11.3f}"
        f"{r['oracle_ate_rmse_m']:>9.6f}"
    )


print()
print("FINAL TEST LOADED: False")
print("DRIVE 0018 LOADED: False")

print()
print(
    "Saved:",
    OUTPUT,
)

print()
print(
    "PASS: drift decomposition performed on VALIDATION only."
)

print("=" * 142)
