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

from src.data.kitti_frame_pairs import (
    KittiFramePairDataset,
)

from src.models.sensor_fusion import (
    SensorFusionMotionModel,
)


R = 6378137.0

CHECKPOINT = (
    ROOT
    / "runs"
    / "final_train"
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

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def wrap_angle(x):

    return math.atan2(
        math.sin(x),
        math.cos(x),
    )


def vehicle_to_global(
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
    lat,
    lon,
    east,
    north,
):

    lat0 = math.radians(
        lat
    )

    lat1 = (
        lat0
        + north / R
    )

    mean_lat = 0.5 * (
        lat0 + lat1
    )

    lon1 = (
        lon
        + math.degrees(
            east
            / (
                R
                * math.cos(mean_lat)
            )
        )
    )

    return (
        math.degrees(lat1),
        lon1,
    )


def position_residual(
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
        R
        * math.radians(
            pred_lat
            - gt_lat
        )
    )

    east = (
        R
        * math.cos(mean_lat)
        * math.radians(
            pred_lon
            - gt_lon
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


best_epoch = int(
    checkpoint["epoch"]
)

stored_val_ate = float(
    checkpoint[
        "best_validation_recursive_ate_rmse_m"
    ]
)

stats = checkpoint[
    "training_statistics"
]


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


use_amp = (
    device.type == "cuda"
)


model = (
    SensorFusionMotionModel()
    .to(device)
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


freeze_record = {

    "checkpoint":
        str(
            CHECKPOINT.resolve()
        ),

    "sha256":
        checkpoint_hash,

    "selected_epoch":
        best_epoch,

    "validation_selection_metric":
        "recursive ATE RMSE",

    "stored_validation_ate_rmse_m":
        stored_val_ate,

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

    "checkpoint_frozen":
        True,
}


(
    RESULTS
    / "final_frozen_checkpoint.json"
).write_text(
    json.dumps(
        freeze_record,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


@torch.no_grad()
def network_predictions(
    dataset,
):

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=use_amp,
    )

    motions = []
    dyaws = []


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
            imu_raw
            - imu_mean
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


        # Heading is propagated from IMU,
        # not predicted by the network.

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


        motions.extend(
            pred.float()
            .cpu()
            .numpy()
            .tolist()
        )

        dyaws.extend(
            dyaw.float()
            .cpu()
            .numpy()
            .tolist()
        )


    return motions, dyaws


def evaluate(
    split,
    filename,
):

    dataset = (
        KittiFramePairDataset(
            split=split,
            root=ROOT,
            camera_size=(160, 96),
            bev_size=128,
            imu_window=5,
            cache_lidar=True,
        )
    )


    motions, dyaws = (
        network_predictions(
            dataset
        )
    )


    output = []

    position_errors = []
    translation_errors = []
    yaw_errors = []

    total_gt_distance = 0.0
    total_pred_distance = 0.0

    previous_key = None
    previous_frame_t1 = None

    state_lat = None
    state_lon = None
    state_yaw = None

    sequence_number = 0


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
            previous_key != key
            or previous_frame_t1 is None
            or frame_t
            != previous_frame_t1
        )


        if new_sequence:

            sequence_number += 1

            # Known once at sequence start.
            state_lat = float(
                row["gt_lat_t"]
            )

            state_lon = float(
                row["gt_lon_t"]
            )

            state_yaw = float(
                row["gt_yaw_t_rad"]
            )


        pred_lat_t = state_lat
        pred_lon_t = state_lon
        pred_yaw_t = state_yaw


        gt_dx = float(
            row["gt_dx_m"]
        )

        gt_dy = float(
            row["gt_dy_m"]
        )


        pred_dx = float(
            motions[i][0]
        )

        pred_dy = float(
            motions[i][1]
        )


        gt_step = math.hypot(
            gt_dx,
            gt_dy,
        )

        pred_step = math.hypot(
            pred_dx,
            pred_dy,
        )


        translation_error = (
            math.hypot(
                pred_dx - gt_dx,
                pred_dy - gt_dy,
            )
        )


        east_move, north_move = (
            vehicle_to_global(
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
            dyaws[i]
        )


        pred_yaw_t1 = (
            wrap_angle(
                pred_yaw_t
                + pred_dyaw
            )
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


        lat_error_deg = (
            pred_lat_t1
            - gt_lat_t1
        )

        lon_error_deg = (
            pred_lon_t1
            - gt_lon_t1
        )


        east_error, north_error = (
            position_residual(
                gt_lat_t1,
                gt_lon_t1,
                pred_lat_t1,
                pred_lon_t1,
            )
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


        total_gt_distance += gt_step
        total_pred_distance += pred_step


        position_errors.append(
            position_error
        )

        translation_errors.append(
            translation_error
        )

        yaw_errors.append(
            abs(
                yaw_error_deg
            )
        )


        output.append({

            "split":
                split,

            "drive":
                drive,

            "sequence_number":
                sequence_number,

            "segment_id":
                segment,

            "frame_t":
                frame_t,

            "frame_t1":
                frame_t1,

            "dt_s":
                float(
                    row["dt_s"]
                ),

            "gt_lat_t":
                gt_lat_t,

            "gt_lon_t":
                gt_lon_t,

            "pred_lat_t":
                pred_lat_t,

            "pred_lon_t":
                pred_lon_t,

            "gt_lat_t1":
                gt_lat_t1,

            "gt_lon_t1":
                gt_lon_t1,

            "pred_lat_t1":
                pred_lat_t1,

            "pred_lon_t1":
                pred_lon_t1,

            "latitude_error_deg":
                lat_error_deg,

            "longitude_error_deg":
                lon_error_deg,

            "east_error_m":
                east_error,

            "north_error_m":
                north_error,

            "frame_position_error_m":
                position_error,

            "ATE_m":
                position_error,

            "gt_dx_m":
                gt_dx,

            "gt_dy_m":
                gt_dy,

            "pred_dx_m":
                pred_dx,

            "pred_dy_m":
                pred_dy,

            "gt_step_m":
                gt_step,

            "pred_step_m":
                pred_step,

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

            "pred_imu_dyaw_rad":
                pred_dyaw,

            "yaw_error_deg":
                yaw_error_deg,

            "cumulative_gt_distance_m":
                total_gt_distance,

            "cumulative_pred_distance_m":
                total_pred_distance,
        })


        # Recursive predicted state.
        state_lat = pred_lat_t1
        state_lon = pred_lon_t1
        state_yaw = pred_yaw_t1

        previous_key = key
        previous_frame_t1 = frame_t1


    output_path = (
        RESULTS
        / filename
    )


    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                output[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            output
        )


    ate_rmse = math.sqrt(
        np.mean(
            np.square(
                position_errors
            )
        )
    )


    translation_rmse = (
        math.sqrt(
            np.mean(
                np.square(
                    translation_errors
                )
            )
        )
    )


    summary = {

        "split":
            split,

        "pairs":
            len(output),

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
                translation_rmse
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

        "total_gt_distance_m":
            float(
                total_gt_distance
            ),

        "total_pred_distance_m":
            float(
                total_pred_distance
            ),

        "distance_ratio":
            float(
                total_pred_distance
                / total_gt_distance
            )
            if total_gt_distance > 0
            else None,

        "final_position_error_m":
            float(
                position_errors[-1]
            ),
    }


    summary_path = (
        RESULTS
        / filename.replace(
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
    print("=" * 88)
    print(
        split.upper()
    )
    print("=" * 88)

    print(
        "Pairs                  :",
        summary["pairs"],
    )

    print(
        "Sequences              :",
        summary["sequences"],
    )

    print(
        "Recursive ATE RMSE     :",
        f"{ate_rmse:.6f} m",
    )

    print(
        "Recursive ATE mean     :",
        f"{summary['recursive_ate_mean_m']:.6f} m",
    )

    print(
        "Recursive ATE median   :",
        f"{summary['recursive_ate_median_m']:.6f} m",
    )

    print(
        "Recursive ATE max      :",
        f"{summary['recursive_ate_max_m']:.6f} m",
    )

    print(
        "Translation RMSE       :",
        f"{translation_rmse:.6f} m",
    )

    print(
        "Yaw MAE                :",
        f"{summary['yaw_mae_deg']:.6f} deg",
    )

    print(
        "Total GT distance      :",
        f"{total_gt_distance:.6f} m",
    )

    print(
        "Total predicted distance:",
        f"{total_pred_distance:.6f} m",
    )

    print(
        "Distance ratio         :",
        f"{summary['distance_ratio']:.6f}",
    )

    print(
        "Final row position error:",
        f"{summary['final_position_error_m']:.6f} m",
    )

    print(
        "Saved CSV              :",
        output_path,
    )


    return summary


print()
print("=" * 88)
print("FINAL FROZEN MODEL")
print("=" * 88)

print(
    "Best epoch          :",
    best_epoch,
)

print(
    "Stored validation ATE:",
    f"{stored_val_ate:.6f} m",
)

print(
    "Checkpoint SHA-256  :",
    checkpoint_hash,
)

print(
    "Device              :",
    device,
)


validation = evaluate(
    "validation",
    "final_validation_frame_predictions.csv",
)


difference = abs(
    validation[
        "recursive_ate_rmse_m"
    ]
    - stored_val_ate
)


print()
print(
    "Validation metric discrepancy:",
    f"{difference:.12e} m",
)


if difference > 1e-4:

    raise RuntimeError(
        "Frozen checkpoint validation metric "
        "was not independently reproduced."
    )


print(
    "PASS: frozen validation metric reproduced."
)


cross_drive = evaluate(
    "cross_drive_evaluation",
    "final_drive0018_frame_predictions.csv",
)


final_summary = {

    "checkpoint_sha256":
        checkpoint_hash,

    "best_epoch":
        best_epoch,

    "validation":
        validation,

    "drive0018_cross_drive":
        cross_drive,

    "status": {
        "checkpoint_frozen_before_this_evaluation":
            True,

        "drive0018_role":
            (
                "cross-drive diagnostic sequence; "
                "not a genuinely untouched final test"
            ),
    },
}


summary_file = (
    RESULTS
    / "final_model_evaluation_summary.json"
)


summary_file.write_text(
    json.dumps(
        final_summary,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


print()
print("=" * 88)
print("FINAL EVALUATION COMPLETE")
print("=" * 88)

print(
    "Frozen epoch:",
    best_epoch,
)

print(
    "Validation ATE:",
    f"{validation['recursive_ate_rmse_m']:.6f} m",
)

print(
    "Drive 0018 cross-drive ATE:",
    f"{cross_drive['recursive_ate_rmse_m']:.6f} m",
)

print()
print(
    "Saved:",
    summary_file,
)
