from pathlib import Path
import csv
import hashlib
import json
import math
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path.cwd()

sys.path.insert(
    0,
    str(ROOT),
)

from src.data.kitti_frame_pairs import KittiFramePairDataset
from src.models.sensor_fusion import SensorFusionMotionModel


EARTH_RADIUS_M = 6378137.0

CHECKPOINT = (
    ROOT
    / "runs"
    / "full_train_100ep"
    / "best_model.pth"
)

RESULTS = ROOT / "results"
RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


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

    mean_lat = 0.5 * (
        lat_rad + next_lat_rad
    )

    next_lon_deg = (
        lon_deg
        + math.degrees(
            east_m
            / (
                EARTH_RADIUS_M
                * math.cos(mean_lat)
            )
        )
    )

    return (
        next_lat_deg,
        next_lon_deg,
    )


def latlon_residual_m(
    gt_lat,
    gt_lon,
    pred_lat,
    pred_lon,
):
    mean_lat = math.radians(
        0.5 * (
            gt_lat + pred_lat
        )
    )

    north = (
        EARTH_RADIUS_M
        * math.radians(
            pred_lat - gt_lat
        )
    )

    east = (
        EARTH_RADIUS_M
        * math.cos(mean_lat)
        * math.radians(
            pred_lon - gt_lon
        )
    )

    return east, north


checkpoint_hash = sha256_file(
    CHECKPOINT
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

best_epoch = checkpoint[
    "epoch"
]

best_val_ate = checkpoint[
    "best_validation_recursive_ate_rmse_m"
]

stats = checkpoint[
    "training_statistics"
]


# ------------------------------------------------------------
# Freeze record
# ------------------------------------------------------------

freeze_record = {
    "checkpoint": str(
        CHECKPOINT.resolve()
    ),

    "sha256": checkpoint_hash,

    "selected_epoch": int(
        best_epoch
    ),

    "selection_metric":
        "validation recursive ATE RMSE",

    "validation_recursive_ate_rmse_m":
        float(best_val_ate),

    "drive_0018_used_for_training":
        False,

    "drive_0018_used_for_model_selection":
        False,

    "note":
        (
            "Checkpoint frozen before "
            "cross-drive evaluation."
        ),
}


freeze_path = (
    RESULTS
    / "frozen_checkpoint.json"
)

freeze_path.write_text(
    json.dumps(
        freeze_record,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

use_amp = (
    device.type == "cuda"
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


imu_mean = torch.tensor(
    stats["imu_mean"],
    dtype=torch.float32,
    device=device,
).view(
    1,
    1,
    6,
)

imu_std = torch.tensor(
    stats["imu_std"],
    dtype=torch.float32,
    device=device,
).view(
    1,
    1,
    6,
)


@torch.no_grad()
def predict_local_motion(
    dataset,
):

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=use_amp,
    )

    predictions = []
    imu_dyaw_values = []

    for batch in loader:

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

        dt = batch[
            "dt"
        ].to(
            device,
            non_blocking=True,
        )

        imu = (
            imu_raw - imu_mean
        ) / imu_std

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

        # Trapezoidal wz integration.
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
                wz_t + wz_t1
            )
            * dt
        )

        predictions.extend(
            pred.float()
            .cpu()
            .numpy()
            .tolist()
        )

        imu_dyaw_values.extend(
            dyaw.float()
            .cpu()
            .numpy()
            .tolist()
        )

    return (
        predictions,
        imu_dyaw_values,
    )


def evaluate_split(
    split,
    output_name,
):

    dataset = KittiFramePairDataset(
        split=split,
        root=ROOT,
        camera_size=(160, 96),
        bev_size=128,
        imu_window=5,
        cache_lidar=True,
    )

    (
        predictions,
        imu_dyaw,
    ) = predict_local_motion(
        dataset
    )

    assert (
        len(predictions)
        == len(dataset.rows)
    )

    output_rows = []

    position_errors = []
    translation_errors = []
    yaw_errors = []

    cumulative_gt = 0.0
    cumulative_pred = 0.0

    previous_key = None
    previous_frame_t1 = None

    pred_lat = None
    pred_lon = None
    pred_yaw = None

    sequence_number = 0


    for i, row in enumerate(
        dataset.rows
    ):

        drive = row["drive"]

        segment_id = int(
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
            segment_id,
        )

        new_sequence = (
            key != previous_key
            or previous_frame_t1 is None
            or frame_t
            != previous_frame_t1
        )


        if new_sequence:

            sequence_number += 1

            pred_lat = float(
                row["gt_lat_t"]
            )

            pred_lon = float(
                row["gt_lon_t"]
            )

            pred_yaw = float(
                row["gt_yaw_t_rad"]
            )

            cumulative_gt = 0.0
            cumulative_pred = 0.0


        pred_lat_t = pred_lat
        pred_lon_t = pred_lon
        pred_yaw_t = pred_yaw


        gt_dx = float(
            row["gt_dx_m"]
        )

        gt_dy = float(
            row["gt_dy_m"]
        )

        pred_dx = float(
            predictions[i][0]
        )

        pred_dy = float(
            predictions[i][1]
        )


        gt_step = math.hypot(
            gt_dx,
            gt_dy,
        )

        pred_step = math.hypot(
            pred_dx,
            pred_dy,
        )


        translation_error = math.hypot(
            pred_dx - gt_dx,
            pred_dy - gt_dy,
        )


        east_move, north_move = (
            vehicle_to_east_north(
                pred_dx,
                pred_dy,
                pred_yaw_t,
            )
        )


        (
            pred_lat_t1,
            pred_lon_t1,
        ) = propagate_latlon(
            pred_lat_t,
            pred_lon_t,
            east_move,
            north_move,
        )


        pred_dyaw = float(
            imu_dyaw[i]
        )

        pred_yaw_t1 = wrap_angle(
            pred_yaw_t
            + pred_dyaw
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

        gt_dyaw = float(
            row["gt_dyaw_rad"]
        )


        latitude_error_deg = (
            pred_lat_t1
            - gt_lat_t1
        )

        longitude_error_deg = (
            pred_lon_t1
            - gt_lon_t1
        )


        (
            east_error,
            north_error,
        ) = latlon_residual_m(
            gt_lat_t1,
            gt_lon_t1,
            pred_lat_t1,
            pred_lon_t1,
        )


        position_error = math.hypot(
            east_error,
            north_error,
        )


        yaw_error_deg = math.degrees(
            wrap_angle(
                pred_yaw_t1
                - gt_yaw_t1
            )
        )


        cumulative_gt += gt_step
        cumulative_pred += pred_step


        position_errors.append(
            position_error
        )

        translation_errors.append(
            translation_error
        )

        yaw_errors.append(
            abs(yaw_error_deg)
        )


        output_rows.append({
            "split": split,
            "drive": drive,
            "sequence_number": sequence_number,
            "segment_id": segment_id,

            "frame_t": frame_t,
            "frame_t1": frame_t1,

            "dt_s": float(
                row["dt_s"]
            ),

            "gt_lat_t": gt_lat_t,
            "gt_lon_t": gt_lon_t,

            "pred_lat_t": pred_lat_t,
            "pred_lon_t": pred_lon_t,

            "gt_lat_t1": gt_lat_t1,
            "gt_lon_t1": gt_lon_t1,

            "pred_lat_t1": pred_lat_t1,
            "pred_lon_t1": pred_lon_t1,

            "latitude_error_deg":
                latitude_error_deg,

            "longitude_error_deg":
                longitude_error_deg,

            "east_error_m":
                east_error,

            "north_error_m":
                north_error,

            "frame_position_error_m":
                position_error,

            "ATE_m":
                position_error,

            "gt_dx_m": gt_dx,
            "gt_dy_m": gt_dy,

            "pred_dx_m": pred_dx,
            "pred_dy_m": pred_dy,

            "gt_step_m": gt_step,
            "pred_step_m": pred_step,

            "frame_translation_error_m":
                translation_error,

            "gt_yaw_t_rad":
                gt_yaw_t,

            "pred_yaw_t_rad":
                pred_yaw_t,

            "gt_yaw_t1_rad":
                gt_yaw_t1,

            "pred_yaw_t1_rad":
                pred_yaw_t1,

            "gt_dyaw_rad":
                gt_dyaw,

            "pred_imu_dyaw_rad":
                pred_dyaw,

            "yaw_error_deg":
                yaw_error_deg,

            "cumulative_gt_distance_m":
                cumulative_gt,

            "cumulative_pred_distance_m":
                cumulative_pred,
        })


        # Recursive predicted state only.
        pred_lat = pred_lat_t1
        pred_lon = pred_lon_t1
        pred_yaw = pred_yaw_t1

        previous_key = key
        previous_frame_t1 = frame_t1


    csv_path = (
        RESULTS
        / output_name
    )

    fieldnames = list(
        output_rows[0].keys()
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )


    ate_rmse = math.sqrt(
        np.mean(
            np.square(
                position_errors
            )
        )
    )

    summary = {
        "split": split,

        "pairs": len(
            output_rows
        ),

        "sequences":
            sequence_number,

        "recursive_ate_rmse_m":
            float(
                ate_rmse
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
            float(
                np.max(
                    position_errors
                )
            ),

        "translation_error_rmse_m":
            float(
                math.sqrt(
                    np.mean(
                        np.square(
                            translation_errors
                        )
                    )
                )
            ),

        "translation_error_mean_m":
            float(
                np.mean(
                    translation_errors
                )
            ),

        "yaw_mae_deg":
            float(
                np.mean(
                    yaw_errors
                )
            ),

        "final_position_error_m":
            float(
                position_errors[-1]
            ),

        "final_gt_distance_m":
            float(
                output_rows[-1][
                    "cumulative_gt_distance_m"
                ]
            ),

        "final_pred_distance_m":
            float(
                output_rows[-1][
                    "cumulative_pred_distance_m"
                ]
            ),
    }


    summary_path = (
        RESULTS
        / output_name.replace(
            ".csv",
            "_summary.json",
        )
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


    print()
    print("=" * 86)
    print(
        split.upper()
    )
    print("=" * 86)

    print(
        "Pairs                    :",
        summary["pairs"],
    )

    print(
        "Sequences                :",
        summary["sequences"],
    )

    print(
        "Recursive ATE RMSE       :",
        f"{summary['recursive_ate_rmse_m']:.6f} m",
    )

    print(
        "Recursive ATE mean       :",
        f"{summary['recursive_ate_mean_m']:.6f} m",
    )

    print(
        "Recursive ATE median     :",
        f"{summary['recursive_ate_median_m']:.6f} m",
    )

    print(
        "Recursive ATE max        :",
        f"{summary['recursive_ate_max_m']:.6f} m",
    )

    print(
        "Translation error RMSE   :",
        f"{summary['translation_error_rmse_m']:.6f} m",
    )

    print(
        "Translation error mean   :",
        f"{summary['translation_error_mean_m']:.6f} m",
    )

    print(
        "Recursive yaw MAE        :",
        f"{summary['yaw_mae_deg']:.6f} deg",
    )

    print(
        "Final position error     :",
        f"{summary['final_position_error_m']:.6f} m",
    )

    print(
        "GT distance              :",
        f"{summary['final_gt_distance_m']:.6f} m",
    )

    print(
        "Predicted distance       :",
        f"{summary['final_pred_distance_m']:.6f} m",
    )

    print(
        "Frame CSV                :",
        csv_path,
    )

    return summary


print()
print("=" * 86)
print("FROZEN CHECKPOINT")
print("=" * 86)

print(
    "Epoch              :",
    best_epoch,
)

print(
    "Validation ATE     :",
    f"{best_val_ate:.6f} m",
)

print(
    "SHA-256            :",
    checkpoint_hash,
)

print(
    "Device             :",
    device,
)

print()
print(
    "Checkpoint is frozen before Drive 0018 evaluation."
)


validation_summary = evaluate_split(
    "validation",
    "validation_frame_predictions.csv",
)


# Verify the independent evaluation reproduces
# the metric used to select the checkpoint.
difference = abs(
    validation_summary[
        "recursive_ate_rmse_m"
    ]
    - float(best_val_ate)
)

print()
print(
    "Validation checkpoint metric discrepancy:",
    f"{difference:.12e} m",
)

if difference > 1e-4:
    raise RuntimeError(
        "Independent validation evaluation "
        "does not reproduce checkpoint metric."
    )

print(
    "PASS: frozen validation metric reproduced."
)


cross_summary = evaluate_split(
    "cross_drive_evaluation",
    "drive0018_frame_predictions.csv",
)


comparison = {
    "frozen_epoch": int(
        best_epoch
    ),

    "checkpoint_sha256":
        checkpoint_hash,

    "validation":
        validation_summary,

    "drive0018_cross_drive":
        cross_summary,

    "scientific_status": {
        "drive0018":
            (
                "cross-drive evaluation only; "
                "not a genuinely untouched final test"
            ),

        "checkpoint_changed_after_0018":
            False,
    },
}


comparison_path = (
    RESULTS
    / "frozen_model_evaluation_summary.json"
)

comparison_path.write_text(
    json.dumps(
        comparison,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


print()
print("=" * 86)
print("EVALUATION COMPLETE")
print("=" * 86)

print(
    "Frozen epoch:",
    best_epoch,
)

print(
    "Validation ATE:",
    f"{validation_summary['recursive_ate_rmse_m']:.6f} m",
)

print(
    "Drive 0018 ATE:",
    f"{cross_summary['recursive_ate_rmse_m']:.6f} m",
)

print()
print(
    "IMPORTANT: do not modify the model "
    "based on Drive 0018 results."
)

print()
print(
    "Saved:",
    comparison_path,
)
