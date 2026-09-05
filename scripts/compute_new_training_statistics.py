from pathlib import Path
import csv
import json
import math

import numpy as np


ROOT = Path.cwd()

MANIFEST = (
    ROOT
    / "results"
    / "new_frame_pair_manifest.csv"
)

OUTPUT = (
    ROOT
    / "results"
    / "training_statistics.json"
)


# ============================================================
# Load TRAINING rows only
# ============================================================

with MANIFEST.open(
    newline="",
    encoding="utf-8-sig",
) as f:

    all_rows = list(
        csv.DictReader(f)
    )


rows = [
    row
    for row in all_rows
    if row["split"] == "train"
]


if len(rows) != 19483:
    raise RuntimeError(
        f"Expected 19483 training pairs, "
        f"found {len(rows)}"
    )


print()
print("=" * 90)
print("NEW TRAINING-ONLY STATISTICS")
print("=" * 90)

print(
    "Training pairs:",
    len(rows),
)


# ============================================================
# Determine segment starts
# ============================================================

segment_start = {}

for row in rows:

    key = (
        row["drive"],
        int(row["segment_id"]),
    )

    frame_t = int(
        row["frame_t"]
    )

    if (
        key not in segment_start
        or frame_t < segment_start[key]
    ):
        segment_start[key] = frame_t


# ============================================================
# Exact IMU input distribution
#
# Same five-frame window used by KittiFramePairDataset.
# OXTS reads are cached so overlapping windows do not cause
# unnecessary disk access.
# ============================================================

oxts_cache = {}

imu_samples = []


def read_imu(
    drive,
    folder,
    frame,
):

    key = (
        drive,
        frame,
    )

    if key in oxts_cache:
        return oxts_cache[key]

    path = (
        Path(folder)
        / "oxts"
        / "data"
        / f"{frame:010d}.txt"
    )

    values = [
        float(x)
        for x in path.read_text(
            encoding="utf-8"
        ).strip().split()
    ]

    if len(values) < 20:
        raise RuntimeError(
            f"OXTS row too short: {path}"
        )

    # af, al, au, wx, wy, wz
    imu = np.asarray(
        values[14:20],
        dtype=np.float64,
    )

    oxts_cache[key] = imu

    return imu


for index, row in enumerate(rows):

    drive = row["drive"]

    segment = int(
        row["segment_id"]
    )

    frame_t1 = int(
        row["frame_t1"]
    )

    start = segment_start[
        (
            drive,
            segment,
        )
    ]

    wanted = [
        max(
            start,
            frame,
        )
        for frame in range(
            frame_t1 - 4,
            frame_t1 + 1,
        )
    ]

    for frame in wanted:

        imu_samples.append(
            read_imu(
                drive,
                row["folder"],
                frame,
            )
        )


imu_array = np.stack(
    imu_samples,
    axis=0,
)


imu_mean = imu_array.mean(
    axis=0
)

imu_std = imu_array.std(
    axis=0
)


# Prevent numerical division problems.
imu_std = np.maximum(
    imu_std,
    1e-8,
)


# ============================================================
# Training motion-regime distribution
# ============================================================

regime_counts = {
    "<0.02": 0,
    "0.02-0.20": 0,
    "0.20-0.50": 0,
    "0.50-1.00": 0,
    ">=1.00": 0,
}


for row in rows:

    step = float(
        row["gt_step_m"]
    )

    if step < 0.02:
        regime_counts["<0.02"] += 1

    elif step < 0.20:
        regime_counts["0.02-0.20"] += 1

    elif step < 0.50:
        regime_counts["0.20-0.50"] += 1

    elif step < 1.00:
        regime_counts["0.50-1.00"] += 1

    else:
        regime_counts[">=1.00"] += 1


# Same inverse-frequency rule used by the existing trainer:
#
#     weight = N / (K * count)
#
# This gives each motion regime equal total contribution
# before the other loss terms are applied.

N = len(rows)
K = len(regime_counts)

motion_weights = {}

for name, count in regime_counts.items():

    if count <= 0:
        raise RuntimeError(
            f"Training regime {name} has no samples."
        )

    motion_weights[name] = (
        N
        / (
            K * count
        )
    )


# ============================================================
# Save
# ============================================================

stats = {
    "training_pairs":
        N,

    "training_drives":
        len({
            row["drive"]
            for row in rows
        }),

    "imu_window":
        5,

    "imu_fields": [
        "af",
        "al",
        "au",
        "wx",
        "wy",
        "wz",
    ],

    "imu_mean":
        imu_mean.tolist(),

    "imu_std":
        imu_std.tolist(),

    "motion_regime_counts":
        regime_counts,

    "motion_regime_weights":
        motion_weights,

    "motion_regime_boundaries_m": [
        0.02,
        0.20,
        0.50,
        1.00,
    ],

    "statistics_source":
        "new_frame_pair_manifest.csv TRAIN split only",

    "validation_used":
        False,

    "diagnostic_0018_used":
        False,

    "final_test_used":
        False,
}


OUTPUT.write_text(
    json.dumps(
        stats,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# Verification
# ============================================================

print()
print("Training drives:", stats["training_drives"])

print()
print("IMU samples used:", len(imu_samples))

print()
print("IMU mean:")
for name, value in zip(
    stats["imu_fields"],
    imu_mean,
):
    print(
        f"  {name:<3} {value: .10f}"
    )

print()
print("IMU std:")
for name, value in zip(
    stats["imu_fields"],
    imu_std,
):
    print(
        f"  {name:<3} {value: .10f}"
    )


print()
print("Motion regimes:")

total_check = 0

for name in regime_counts:

    count = regime_counts[name]

    total_check += count

    pct = (
        100.0
        * count
        / N
    )

    weight = motion_weights[
        name
    ]

    print(
        f"  {name:<10} "
        f"count={count:5d} "
        f"({pct:6.2f}%) "
        f"weight={weight:8.4f}"
    )


if total_check != N:
    raise RuntimeError(
        "Motion-regime counts do not "
        "sum to training-pair count."
    )


# Check weighted totals are equal.
weighted_totals = {
    name:
        regime_counts[name]
        * motion_weights[name]
    for name in regime_counts
}


values = list(
    weighted_totals.values()
)

if (
    max(values)
    - min(values)
    > 1e-6
):
    raise RuntimeError(
        "Inverse-frequency weighting verification failed."
    )


print()
print("Weighted contribution per regime:")
for name, value in weighted_totals.items():
    print(
        f"  {name:<10} {value:.3f}"
    )


print()
print(
    "Validation used       :",
    stats["validation_used"],
)

print(
    "Diagnostic 0018 used  :",
    stats["diagnostic_0018_used"],
)

print(
    "Final test used       :",
    stats["final_test_used"],
)


print()
print("Saved:", OUTPUT)

print()
print(
    "PASS: statistics derived from "
    "TRAINING DATA ONLY."
)

print("=" * 90)
