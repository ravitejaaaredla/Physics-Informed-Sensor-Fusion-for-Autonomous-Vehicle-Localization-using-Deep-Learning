from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt


ROOT = Path.cwd()

PREDICTIONS = (
    ROOT
    / "results"
    / "final_test_frame_predictions.csv"
)

METRICS = (
    ROOT
    / "results"
    / "final_test_drive_metrics.csv"
)

OUT = (
    ROOT
    / "results"
    / "thesis_package"
    / "figures"
    / "final_test_trajectories"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


EARTH_RADIUS = 6378137.0


# ============================================================
# Load locked predictions / metrics
# ============================================================

with PREDICTIONS.open(
    newline="",
    encoding="utf-8-sig",
) as f:

    rows = list(
        csv.DictReader(f)
    )


with METRICS.open(
    newline="",
    encoding="utf-8-sig",
) as f:

    metrics = {
        row["drive"]: row
        for row in csv.DictReader(f)
    }


drives = sorted({
    row["drive"]
    for row in rows
})


def local_en(
    lat,
    lon,
    lat0,
    lon0,
):

    north = (
        EARTH_RADIUS
        * math.radians(
            lat - lat0
        )
    )

    mean_lat = math.radians(
        0.5
        * (
            lat + lat0
        )
    )

    east = (
        EARTH_RADIUS
        * math.cos(
            mean_lat
        )
        * math.radians(
            lon - lon0
        )
    )

    return (
        east,
        north,
    )


def short_drive_name(
    drive,
):

    return (
        drive
        .replace(
            "2011_09_26_drive_",
            ""
        )
        .replace(
            "2011_09_30_drive_",
            ""
        )
        .replace(
            "2011_10_03_drive_",
            ""
        )
        .replace(
            "_sync",
            ""
        )
    )


print()
print("=" * 100)
print("THESIS FINAL-TEST TRAJECTORY FIGURES")
print("=" * 100)


for drive in drives:

    drive_rows = [
        row
        for row in rows
        if row["drive"] == drive
    ]


    first = drive_rows[0]

    lat0 = float(
        first["gt_lat_t"]
    )

    lon0 = float(
        first["gt_lon_t"]
    )


    gt_east = [0.0]
    gt_north = [0.0]

    pred_east = [0.0]
    pred_north = [0.0]


    for row in drive_rows:

        e, n = local_en(
            float(
                row["gt_lat_t1"]
            ),
            float(
                row["gt_lon_t1"]
            ),
            lat0,
            lon0,
        )

        gt_east.append(e)
        gt_north.append(n)


        e, n = local_en(
            float(
                row["pred_lat_t1"]
            ),
            float(
                row["pred_lon_t1"]
            ),
            lat0,
            lon0,
        )

        pred_east.append(e)
        pred_north.append(n)


    m = metrics[
        drive
    ]


    ate = float(
        m[
            "recursive_ate_rmse_m"
        ]
    )

    final_error = float(
        m[
            "final_position_error_m"
        ]
    )

    gt_distance = float(
        m[
            "gt_distance_m"
        ]
    )

    pred_distance = float(
        m[
            "pred_distance_m"
        ]
    )

    ratio = float(
        m[
            "pred_gt_distance_ratio"
        ]
    )


    # --------------------------------------------------------
    # Choose a wider figure when the trajectory is mostly
    # horizontal, while preserving equal metric axis scaling.
    # --------------------------------------------------------

    all_east = (
        gt_east
        + pred_east
    )

    all_north = (
        gt_north
        + pred_north
    )

    east_span = max(
        all_east
    ) - min(
        all_east
    )

    north_span = max(
        all_north
    ) - min(
        all_north
    )

    aspect_ratio = (
        east_span
        / max(
            north_span,
            1.0
        )
    )


    if aspect_ratio > 1.7:

        figsize = (
            11.0,
            6.0,
        )

    elif aspect_ratio < 0.65:

        figsize = (
            7.0,
            8.5,
        )

    else:

        figsize = (
            8.5,
            7.0,
        )


    fig, ax = plt.subplots(
        figsize=figsize
    )


    ax.plot(
        gt_east,
        gt_north,
        linewidth=2.2,
        label="Ground Truth",
    )

    ax.plot(
        pred_east,
        pred_north,
        linewidth=2.0,
        label="Predicted",
    )


    # Initial global anchor.
    ax.scatter(
        gt_east[0],
        gt_north[0],
        s=65,
        marker="o",
        label="Initial Anchor",
        zorder=5,
    )


    # Final locations.
    ax.scatter(
        gt_east[-1],
        gt_north[-1],
        s=55,
        marker="x",
        label="GT End",
        zorder=5,
    )

    ax.scatter(
        pred_east[-1],
        pred_north[-1],
        s=55,
        marker="+",
        label="Predicted End",
        zorder=5,
    )


    short = short_drive_name(
        drive
    )


    ax.set_title(
        f"KITTI Drive {short} — "
        "Final-Test Recursive Trajectory"
    )

    ax.set_xlabel(
        "East displacement (m)"
    )

    ax.set_ylabel(
        "North displacement (m)"
    )


    # Preserve physical geometry.
    ax.set_aspect(
        "equal",
        adjustable="datalim",
    )


    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="best"
    )


    metric_text = (
        f"GT distance: {gt_distance:.2f} m\n"
        f"Predicted distance: {pred_distance:.2f} m\n"
        f"Pred/GT ratio: {ratio:.3f}\n"
        f"ATE RMSE: {ate:.3f} m\n"
        f"Final error: {final_error:.3f} m"
    )


    ax.text(
        0.02,
        0.98,
        metric_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle":
                "round,pad=0.45",

            "facecolor":
                "white",

            "alpha":
                0.85,
        },
    )


    fig.tight_layout()


    png_path = (
        OUT
        / f"final_test_drive_{short}_trajectory.png"
    )

    pdf_path = (
        OUT
        / f"final_test_drive_{short}_trajectory.pdf"
    )


    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )


    plt.close(fig)


    print(
        "Saved:",
        png_path.name
    )

    print(
        "Saved:",
        pdf_path.name
    )


print()
print(
    "Output folder:",
    OUT
)

print()
print(
    "Predictions modified : False"
)

print(
    "Model modified       : False"
)

print()
print(
    "PASS: thesis final-test trajectory "
    "figures created from locked predictions."
)

print("=" * 100)
