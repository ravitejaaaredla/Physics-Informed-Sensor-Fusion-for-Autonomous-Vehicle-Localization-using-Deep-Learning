from pathlib import Path
import csv
import math
import numpy as np


ROOT = Path.cwd()
DATA = ROOT / "data"
RESULTS = ROOT / "results"

DRIVE_SPLIT = RESULTS / "new_drive_level_split.csv"
OUTPUT = RESULTS / "new_frame_pair_manifest.csv"

EARTH_R = 6378137.0


def read_oxts(path):
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

    return values


def wrap_angle(x):
    return (
        x + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi


def displacement(lat0, lon0, yaw0, lat1, lon1):

    lat0_r = math.radians(lat0)
    lat1_r = math.radians(lat1)

    lon0_r = math.radians(lon0)
    lon1_r = math.radians(lon1)

    mean_lat = 0.5 * (
        lat0_r + lat1_r
    )

    north = (
        EARTH_R
        * (lat1_r - lat0_r)
    )

    east = (
        EARTH_R
        * math.cos(mean_lat)
        * (lon1_r - lon0_r)
    )

    c = math.cos(yaw0)
    s = math.sin(yaw0)

    # Global East/North -> local vehicle frame.
    dx = (
        c * east
        + s * north
    )

    dy = (
        -s * east
        + c * north
    )

    return dx, dy


def regime(step):

    if step < 0.02:
        return "stationary"

    if step < 0.20:
        return "very_low"

    if step < 0.50:
        return "low"

    if step < 1.00:
        return "medium"

    return "high"


def read_timestamps(path):

    if not path.exists():
        raise RuntimeError(
            f"Missing timestamps file: {path}"
        )

    lines = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    return [
        np.datetime64(
            line.replace(" ", "T")
        )
        for line in lines
    ]


# ------------------------------------------------------------
# Load frozen drive-level split
# ------------------------------------------------------------

with DRIVE_SPLIT.open(
    newline="",
    encoding="utf-8-sig",
) as f:

    split_rows = list(
        csv.DictReader(f)
    )


split_by_drive = {
    row["drive"]: row["split"]
    for row in split_rows
}


if len(split_by_drive) != 23:
    raise RuntimeError(
        f"Expected 23 unique drives, "
        f"found {len(split_by_drive)}"
    )


# ------------------------------------------------------------
# Find each physical drive exactly once
# ------------------------------------------------------------

drive_dirs = {}

for drive_name in split_by_drive:

    matches = [
        p
        for p in DATA.rglob(drive_name)
        if (
            p.is_dir()
            and
            (p / "image_02" / "data").is_dir()
            and
            (p / "velodyne_points" / "data").is_dir()
            and
            (p / "oxts" / "data").is_dir()
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"{drive_name}: expected exactly "
            f"one physical directory, "
            f"found {len(matches)}"
        )

    drive_dirs[
        drive_name
    ] = matches[0]


# ------------------------------------------------------------
# Build synchronized contiguous frame pairs
# ------------------------------------------------------------

manifest = []

for drive_name in sorted(drive_dirs):

    drive = drive_dirs[
        drive_name
    ]

    split = split_by_drive[
        drive_name
    ]

    cam_dir = (
        drive
        / "image_02"
        / "data"
    )

    lidar_dir = (
        drive
        / "velodyne_points"
        / "data"
    )

    oxts_dir = (
        drive
        / "oxts"
        / "data"
    )

    timestamps = read_timestamps(
        drive
        / "oxts"
        / "timestamps.txt"
    )

    camera_ids = {
        int(p.stem)
        for p in cam_dir.glob("*.png")
    }

    lidar_ids = {
        int(p.stem)
        for p in lidar_dir.glob("*.bin")
    }

    oxts_ids = {
        int(p.stem)
        for p in oxts_dir.glob("*.txt")
    }

    common = sorted(
        camera_ids
        & lidar_ids
        & oxts_ids
    )

    pairs = [
        (a, b)
        for a, b in zip(
            common[:-1],
            common[1:],
        )
        if b == a + 1
    ]

    segment_id = -1
    previous_t1 = None

    for frame_t, frame_t1 in pairs:

        if (
            previous_t1 is None
            or frame_t != previous_t1
        ):
            segment_id += 1

        previous_t1 = frame_t1

        if frame_t1 >= len(timestamps):
            raise RuntimeError(
                f"Timestamp index out of range: "
                f"{drive_name} frame {frame_t1}"
            )

        o0 = read_oxts(
            oxts_dir
            / f"{frame_t:010d}.txt"
        )

        o1 = read_oxts(
            oxts_dir
            / f"{frame_t1:010d}.txt"
        )

        lat0 = o0[0]
        lon0 = o0[1]
        yaw0 = o0[5]

        lat1 = o1[0]
        lon1 = o1[1]
        yaw1 = o1[5]

        dx, dy = displacement(
            lat0,
            lon0,
            yaw0,
            lat1,
            lon1,
        )

        step = math.hypot(
            dx,
            dy,
        )

        dyaw = wrap_angle(
            yaw1 - yaw0
        )

        dt = float(
            (
                timestamps[frame_t1]
                - timestamps[frame_t]
            )
            / np.timedelta64(
                1,
                "s",
            )
        )

        if not (
            0.0 < dt < 1.0
        ):
            raise RuntimeError(
                f"Invalid dt={dt} "
                f"for {drive_name} "
                f"{frame_t}->{frame_t1}"
            )

        manifest.append({
            "drive":
                drive_name,

            "split":
                split,

            "segment_id":
                segment_id,

            "frame_t":
                frame_t,

            "frame_t1":
                frame_t1,

            "dt_s":
                dt,

            "gt_lat_t":
                lat0,

            "gt_lon_t":
                lon0,

            "gt_lat_t1":
                lat1,

            "gt_lon_t1":
                lon1,

            "gt_yaw_t_rad":
                yaw0,

            "gt_yaw_t1_rad":
                yaw1,

            "gt_dyaw_rad":
                dyaw,

            "gt_dx_m":
                dx,

            "gt_dy_m":
                dy,

            "gt_step_m":
                step,

            "motion_regime":
                regime(step),

            "folder":
                str(
                    drive.resolve()
                ),
        })


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

fieldnames = list(
    manifest[0].keys()
)

with OUTPUT.open(
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
        manifest
    )


# ------------------------------------------------------------
# Verification report
# ------------------------------------------------------------

print()
print("=" * 100)
print("NEW FRAME-PAIR MANIFEST")
print("=" * 100)

print(
    "Total rows:",
    len(manifest),
)

print()

splits = [
    "train",
    "validation",
    "diagnostic_0018",
    "final_test",
]

for split in splits:

    subset = [
        row
        for row in manifest
        if row["split"] == split
    ]

    drives = sorted({
        row["drive"]
        for row in subset
    })

    segments = {
        (
            row["drive"],
            row["segment_id"],
        )
        for row in subset
    }

    print(
        f"{split:<17} "
        f"drives={len(drives):2d} "
        f"pairs={len(subset):5d} "
        f"segments={len(segments):2d}"
    )


print()
print("Drive 0009 segment check:")

d0009 = [
    row
    for row in manifest
    if row["drive"]
    == "2011_09_26_drive_0009_sync"
]

segments_0009 = sorted({
    row["segment_id"]
    for row in d0009
})

print(
    "0009 segments:",
    segments_0009,
)

print(
    "0009 pairs   :",
    len(d0009),
)


if len(manifest) != 24070:
    raise RuntimeError(
        f"Expected 24070 total pairs, "
        f"found {len(manifest)}"
    )

if len(segments_0009) != 2:
    raise RuntimeError(
        "Drive 0009 should have exactly "
        "two contiguous segments."
    )


print()
print(
    "Saved:",
    OUTPUT,
)

print()
print(
    "PASS: manifest contains all synchronized "
    "contiguous pairs with drive-level splits."
)

print("=" * 100)
