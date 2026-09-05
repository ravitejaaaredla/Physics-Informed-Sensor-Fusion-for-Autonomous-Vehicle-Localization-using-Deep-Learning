from pathlib import Path
import csv
import math
import re
import statistics


ROOT = Path.cwd()
DATA_ROOT = ROOT / "data"
RESULTS = ROOT / "results"

CSV_PATH = RESULTS / "frame_pair_audit.csv"

EARTH_RADIUS_M = 6378137.0


def wrap_angle(x):
    return math.atan2(
        math.sin(x),
        math.cos(x),
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
    lat_rad = math.radians(lat_deg)

    next_lat_rad = (
        lat_rad
        + north_m / EARTH_RADIUS_M
    )

    next_lat_deg = math.degrees(
        next_lat_rad
    )

    # Symmetric short-range equirectangular
    # conversion using mean latitude.
    mean_lat_rad = 0.5 * (
        lat_rad + next_lat_rad
    )

    cos_lat = math.cos(
        mean_lat_rad
    )

    if abs(cos_lat) < 1e-12:
        raise RuntimeError(
            "Longitude propagation unstable "
            "near geographic pole."
        )

    next_lon_deg = (
        lon_deg
        + math.degrees(
            east_m
            / (
                EARTH_RADIUS_M
                * cos_lat
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

    return math.hypot(
        east,
        north,
    )


def read_oxts(path):
    return [
        float(x)
        for x in path.read_text(
            encoding="utf-8"
        ).strip().split()
    ]


drive_pattern = re.compile(
    r"2011_09_26_drive_(\d{4})_sync$"
)

drive_dirs = {}

for path in DATA_ROOT.rglob(
    "2011_09_26_drive_*_sync"
):
    if not path.is_dir():
        continue

    m = drive_pattern.search(
        path.name
    )

    if m:
        drive_dirs[
            m.group(1)
        ] = path


with CSV_PATH.open(
    newline="",
    encoding="utf-8",
) as f:
    rows = list(
        csv.DictReader(f)
    )


position_errors = []
lat_errors = []
lon_errors = []

gyro_errors_deg = []
gyro_by_drive = {}

print()
print("=" * 78)
print("PHYSICS ROUND-TRIP VERIFICATION")
print("=" * 78)


for row in rows:

    drive = row["drive"]

    frame_t = int(
        row["frame_t"]
    )

    frame_t1 = int(
        row["frame_t1"]
    )

    lat_t = float(
        row["gt_lat_t"]
    )

    lon_t = float(
        row["gt_lon_t"]
    )

    gt_lat_t1 = float(
        row["gt_lat_t1"]
    )

    gt_lon_t1 = float(
        row["gt_lon_t1"]
    )

    yaw_t = float(
        row["gt_yaw_t_rad"]
    )

    dx = float(
        row["gt_dx_m"]
    )

    dy = float(
        row["gt_dy_m"]
    )

    east, north = (
        vehicle_to_east_north(
            dx,
            dy,
            yaw_t,
        )
    )

    (
        pred_lat_t1,
        pred_lon_t1,
    ) = propagate_latlon(
        lat_t,
        lon_t,
        east,
        north,
    )

    lat_err = abs(
        pred_lat_t1
        - gt_lat_t1
    )

    lon_err = abs(
        pred_lon_t1
        - gt_lon_t1
    )

    pos_err = latlon_error_m(
        gt_lat_t1,
        gt_lon_t1,
        pred_lat_t1,
        pred_lon_t1,
    )

    lat_errors.append(
        lat_err
    )

    lon_errors.append(
        lon_err
    )

    position_errors.append(
        pos_err
    )

    # --------------------------------------------------
    # Raw gyroscope diagnostic
    # KITTI OXTS fields:
    # 14 af
    # 15 al
    # 16 au
    # 17 wx
    # 18 wy
    # 19 wz
    # --------------------------------------------------

    drive_dir = drive_dirs[
        drive
    ]

    oxts_dir = (
        drive_dir
        / "oxts"
        / "data"
    )

    a = read_oxts(
        oxts_dir
        / f"{frame_t:010d}.txt"
    )

    b = read_oxts(
        oxts_dir
        / f"{frame_t1:010d}.txt"
    )

    if (
        len(a) > 19
        and len(b) > 19
        and row["dt_s"]
    ):
        dt = float(
            row["dt_s"]
        )

        wz_t = a[19]
        wz_t1 = b[19]

        imu_dyaw = (
            0.5
            * (
                wz_t + wz_t1
            )
            * dt
        )

        gt_dyaw = float(
            row["gt_dyaw_rad"]
        )

        err = abs(
            math.degrees(
                wrap_angle(
                    imu_dyaw
                    - gt_dyaw
                )
            )
        )

        gyro_errors_deg.append(
            err
        )

        gyro_by_drive.setdefault(
            drive,
            []
        ).append(
            err
        )


print()
print("GEOGRAPHIC RECONSTRUCTION")
print("-" * 78)

print(
    f"Pairs checked               : "
    f"{len(position_errors)}"
)

print(
    f"Mean position discrepancy   : "
    f"{statistics.mean(position_errors):.12e} m"
)

print(
    f"Max position discrepancy    : "
    f"{max(position_errors):.12e} m"
)

print(
    f"Max latitude discrepancy    : "
    f"{max(lat_errors):.12e} deg"
)

print(
    f"Max longitude discrepancy   : "
    f"{max(lon_errors):.12e} deg"
)


print()
print("RAW IMU YAW-INCREMENT DIAGNOSTIC")
print("-" * 78)

for drive in sorted(
    gyro_by_drive
):
    values = gyro_by_drive[
        drive
    ]

    print(
        f"Drive {drive}: "
        f"n={len(values):4d} "
        f"MAE={statistics.mean(values):.6f} deg "
        f"median={statistics.median(values):.6f} deg "
        f"max={max(values):.6f} deg"
    )

print()

if gyro_errors_deg:
    print(
        f"Overall raw gyro yaw MAE     : "
        f"{statistics.mean(gyro_errors_deg):.6f} deg"
    )


print()
print("=" * 78)

if max(position_errors) < 1e-5:
    print(
        "PASS: FRAME-TO-FRAME GEOGRAPHIC PHYSICS VERIFIED"
    )
else:
    print(
        "FAIL: GEOGRAPHIC RECONSTRUCTION NEEDS INVESTIGATION"
    )

print("=" * 78)


report = RESULTS / "physics_verification.txt"

report.write_text(
    "\n".join([
        "PHYSICS VERIFICATION",
        "====================",
        "",
        f"Pairs checked: {len(position_errors)}",
        (
            "Mean reconstruction error [m]: "
            f"{statistics.mean(position_errors):.12e}"
        ),
        (
            "Max reconstruction error [m]: "
            f"{max(position_errors):.12e}"
        ),
        (
            "Raw gyro yaw MAE [deg]: "
            f"{statistics.mean(gyro_errors_deg):.9f}"
            if gyro_errors_deg
            else "Raw gyro yaw MAE: unavailable"
        ),
    ]) + "\n",
    encoding="utf-8",
)

print()
print(
    "Saved:",
    report
)
