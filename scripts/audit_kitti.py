from pathlib import Path
from datetime import datetime
import csv
import math
import re
import statistics


ROOT = Path.cwd()
DATA_ROOT = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_M = 6378137.0


def frame_index(path):
    return int(path.stem)


def compact_ranges(values):
    values = sorted(set(values))

    if not values:
        return "none"

    ranges = []
    start = prev = values[0]

    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue

        ranges.append(
            str(start) if start == prev
            else f"{start}-{prev}"
        )

        start = prev = value

    ranges.append(
        str(start) if start == prev
        else f"{start}-{prev}"
    )

    return ", ".join(ranges)


def read_oxts(path):
    values = [
        float(x)
        for x in path.read_text(
            encoding="utf-8"
        ).strip().split()
    ]

    if len(values) < 6:
        raise ValueError(
            f"Invalid OXTS row: {path}"
        )

    return {
        "lat": values[0],
        "lon": values[1],
        "alt": values[2],
        "roll": values[3],
        "pitch": values[4],
        "yaw": values[5],
        "values": values,
    }


def read_timestamps(path):
    if not path.exists():
        return []

    stamps = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        text = line.strip()

        if not text:
            continue

        if "." in text:
            base, frac = text.split(".", 1)

            dt = datetime.strptime(
                base,
                "%Y-%m-%d %H:%M:%S",
            )

            fraction_s = float(
                "0." + frac
            )

        else:
            dt = datetime.strptime(
                text,
                "%Y-%m-%d %H:%M:%S",
            )

            fraction_s = 0.0

        stamps.append(
            dt.timestamp() + fraction_s
        )

    return stamps


def wrap_angle(angle):
    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


def latlon_to_east_north(
    lat0_deg,
    lon0_deg,
    lat1_deg,
    lon1_deg,
):
    lat0 = math.radians(lat0_deg)
    lat1 = math.radians(lat1_deg)

    dlat = math.radians(
        lat1_deg - lat0_deg
    )

    dlon = math.radians(
        lon1_deg - lon0_deg
    )

    mean_lat = 0.5 * (
        lat0 + lat1
    )

    north = (
        EARTH_RADIUS_M * dlat
    )

    east = (
        EARTH_RADIUS_M
        * math.cos(mean_lat)
        * dlon
    )

    return east, north


def east_north_to_vehicle(
    east,
    north,
    yaw,
):
    c = math.cos(yaw)
    s = math.sin(yaw)

    dx = (
        c * east
        + s * north
    )

    dy = (
        -s * east
        + c * north
    )

    return dx, dy


def contiguous_segments(indices):
    indices = sorted(indices)

    if not indices:
        return []

    segments = []

    start = prev = indices[0]

    for idx in indices[1:]:

        if idx == prev + 1:
            prev = idx
            continue

        segments.append(
            (start, prev)
        )

        start = prev = idx

    segments.append(
        (start, prev)
    )

    return segments


drive_pattern = re.compile(
    r"2011_09_26_drive_(\d{4})_sync$"
)

drive_dirs = []

for path in DATA_ROOT.rglob(
    "2011_09_26_drive_*_sync"
):
    if (
        path.is_dir()
        and drive_pattern.search(path.name)
    ):
        drive_dirs.append(path)

drive_dirs = sorted(
    set(drive_dirs),
    key=lambda p: p.name,
)

if not drive_dirs:
    raise RuntimeError(
        "No KITTI synchronized drives found."
    )


all_rows = []
summary_lines = []

print()
print("=" * 78)
print("KITTI FRAME-TO-FRAME AUDIT")
print("=" * 78)

for drive_dir in drive_dirs:

    match = drive_pattern.search(
        drive_dir.name
    )

    drive = match.group(1)

    camera_dir = (
        drive_dir
        / "image_02"
        / "data"
    )

    lidar_dir = (
        drive_dir
        / "velodyne_points"
        / "data"
    )

    oxts_dir = (
        drive_dir
        / "oxts"
        / "data"
    )

    timestamp_file = (
        drive_dir
        / "oxts"
        / "timestamps.txt"
    )

    camera = {
        frame_index(p): p
        for p in camera_dir.glob("*.png")
    }

    lidar = {
        frame_index(p): p
        for p in lidar_dir.glob("*.bin")
    }

    oxts = {
        frame_index(p): p
        for p in oxts_dir.glob("*.txt")
    }

    timestamps = read_timestamps(
        timestamp_file
    )

    camera_idx = set(camera)
    lidar_idx = set(lidar)
    oxts_idx = set(oxts)

    union_idx = (
        camera_idx
        | lidar_idx
        | oxts_idx
    )

    common_idx = (
        camera_idx
        & lidar_idx
        & oxts_idx
    )

    missing_camera = (
        union_idx - camera_idx
    )

    missing_lidar = (
        union_idx - lidar_idx
    )

    missing_oxts = (
        union_idx - oxts_idx
    )

    segments = contiguous_segments(
        common_idx
    )

    valid_pair_count = sum(
        max(0, end - start)
        for start, end in segments
    )

    print()
    print(f"Drive {drive}")
    print("-" * 78)

    print(
        f"Camera frames : {len(camera_idx)}"
    )

    print(
        f"LiDAR frames  : {len(lidar_idx)}"
    )

    print(
        f"OXTS frames   : {len(oxts_idx)}"
    )

    print(
        f"Common frames : {len(common_idx)}"
    )

    print(
        f"Valid pairs   : {valid_pair_count}"
    )

    print(
        "Missing camera:",
        compact_ranges(
            missing_camera
        ),
    )

    print(
        "Missing LiDAR :",
        compact_ranges(
            missing_lidar
        ),
    )

    print(
        "Missing OXTS  :",
        compact_ranges(
            missing_oxts
        ),
    )

    print(
        f"Segments      : {len(segments)}"
    )

    for segment_id, (
        start,
        end,
    ) in enumerate(
        segments,
        start=1,
    ):
        print(
            f"  segment {segment_id}: "
            f"{start} -> {end} "
            f"({end-start} frame pairs)"
        )

    drive_steps = []
    drive_dt = []
    drive_speeds = []

    for segment_id, (
        start,
        end,
    ) in enumerate(
        segments,
        start=1,
    ):

        for t in range(
            start,
            end,
        ):
            t1 = t + 1

            if (
                t not in common_idx
                or t1 not in common_idx
            ):
                continue

            a = read_oxts(
                oxts[t]
            )

            b = read_oxts(
                oxts[t1]
            )

            east_m, north_m = (
                latlon_to_east_north(
                    a["lat"],
                    a["lon"],
                    b["lat"],
                    b["lon"],
                )
            )

            dx_m, dy_m = (
                east_north_to_vehicle(
                    east_m,
                    north_m,
                    a["yaw"],
                )
            )

            step_m = math.hypot(
                dx_m,
                dy_m,
            )

            dyaw_rad = wrap_angle(
                b["yaw"] - a["yaw"]
            )

            dt_s = None

            if (
                t < len(timestamps)
                and t1 < len(timestamps)
            ):
                dt_s = (
                    timestamps[t1]
                    - timestamps[t]
                )

            speed_mps = None

            if (
                dt_s is not None
                and dt_s > 0
            ):
                speed_mps = (
                    step_m / dt_s
                )

            row = {
                "drive": drive,
                "segment_id": segment_id,
                "frame_t": t,
                "frame_t1": t1,

                "dt_s": (
                    ""
                    if dt_s is None
                    else f"{dt_s:.9f}"
                ),

                "gt_lat_t": (
                    f"{a['lat']:.12f}"
                ),

                "gt_lon_t": (
                    f"{a['lon']:.12f}"
                ),

                "gt_lat_t1": (
                    f"{b['lat']:.12f}"
                ),

                "gt_lon_t1": (
                    f"{b['lon']:.12f}"
                ),

                "delta_lat_deg": (
                    f"{b['lat']-a['lat']:.12e}"
                ),

                "delta_lon_deg": (
                    f"{b['lon']-a['lon']:.12e}"
                ),

                "gt_yaw_t_rad": (
                    f"{a['yaw']:.12f}"
                ),

                "gt_yaw_t1_rad": (
                    f"{b['yaw']:.12f}"
                ),

                "gt_dyaw_rad": (
                    f"{dyaw_rad:.12f}"
                ),

                "gt_dyaw_deg": (
                    f"{math.degrees(dyaw_rad):.9f}"
                ),

                "east_m": (
                    f"{east_m:.9f}"
                ),

                "north_m": (
                    f"{north_m:.9f}"
                ),

                "gt_dx_m": (
                    f"{dx_m:.9f}"
                ),

                "gt_dy_m": (
                    f"{dy_m:.9f}"
                ),

                "gt_step_m": (
                    f"{step_m:.9f}"
                ),

                "gt_speed_mps": (
                    ""
                    if speed_mps is None
                    else f"{speed_mps:.9f}"
                ),
            }

            all_rows.append(row)

            drive_steps.append(
                step_m
            )

            if dt_s is not None:
                drive_dt.append(
                    dt_s
                )

            if speed_mps is not None:
                drive_speeds.append(
                    speed_mps
                )

    if drive_steps:
        print(
            f"Mean step     : "
            f"{statistics.mean(drive_steps):.6f} m"
        )

        print(
            f"Median step   : "
            f"{statistics.median(drive_steps):.6f} m"
        )

        print(
            f"Max step      : "
            f"{max(drive_steps):.6f} m"
        )

    if drive_dt:
        print(
            f"Mean dt       : "
            f"{statistics.mean(drive_dt):.6f} s"
        )

    summary_lines.extend([
        f"Drive {drive}",
        f"Camera frames : {len(camera_idx)}",
        f"LiDAR frames  : {len(lidar_idx)}",
        f"OXTS frames   : {len(oxts_idx)}",
        f"Common frames : {len(common_idx)}",
        f"Valid pairs   : {valid_pair_count}",
        (
            "Missing camera: "
            + compact_ranges(
                missing_camera
            )
        ),
        (
            "Missing LiDAR : "
            + compact_ranges(
                missing_lidar
            )
        ),
        (
            "Missing OXTS  : "
            + compact_ranges(
                missing_oxts
            )
        ),
        f"Segments      : {segments}",
        "",
    ])


csv_path = (
    RESULTS
    / "frame_pair_audit.csv"
)

fieldnames = [
    "drive",
    "segment_id",
    "frame_t",
    "frame_t1",
    "dt_s",

    "gt_lat_t",
    "gt_lon_t",

    "gt_lat_t1",
    "gt_lon_t1",

    "delta_lat_deg",
    "delta_lon_deg",

    "gt_yaw_t_rad",
    "gt_yaw_t1_rad",

    "gt_dyaw_rad",
    "gt_dyaw_deg",

    "east_m",
    "north_m",

    "gt_dx_m",
    "gt_dy_m",

    "gt_step_m",
    "gt_speed_mps",
]

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
        all_rows
    )


summary_path = (
    RESULTS
    / "data_audit_summary.txt"
)

summary_path.write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)


print()
print("=" * 78)
print("GLOBAL AUDIT")
print("=" * 78)

print(
    "Drives found :",
    len(drive_dirs),
)

print(
    "Total pairs  :",
    len(all_rows),
)

print(
    "CSV saved    :",
    csv_path,
)

print(
    "Summary saved:",
    summary_path,
)

print()
print("FIRST 5 FRAME PAIRS")
print("-" * 78)

for row in all_rows[:5]:

    print(
        f"drive={row['drive']} "
        f"{row['frame_t']}->{row['frame_t1']} "
        f"dLat={row['delta_lat_deg']} "
        f"dLon={row['delta_lon_deg']} "
        f"dx={row['gt_dx_m']}m "
        f"dy={row['gt_dy_m']}m "
        f"step={row['gt_step_m']}m "
        f"dyaw={row['gt_dyaw_deg']}deg"
    )

print()
print("FRAME-TO-FRAME AUDIT COMPLETE")
