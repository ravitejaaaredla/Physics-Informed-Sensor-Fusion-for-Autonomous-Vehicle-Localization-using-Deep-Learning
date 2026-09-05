from pathlib import Path
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

from scripts.publication_style import (
    apply_publication_style,
    clean_axis,
    subtle_grid,
    save_publication_figure,
    GT_COLOR,
    PRED_COLOR,
    GRAY_DARK,
    GRAY_MID,
    GRAY_LIGHT,
)

apply_publication_style()

ROOT = Path.cwd()
RESULTS = ROOT / "results"

OUT = (
    RESULTS
    / "thesis_package"
    / "figures"
    / "PUBLICATION_12_THESIS_FIGURES"
)

OUT.mkdir(parents=True, exist_ok=True)

PRED_FILE = RESULTS / "final_test_frame_predictions.csv"
METRIC_FILE = RESULTS / "final_test_drive_metrics.csv"
DRIFT_FILE = RESULTS / "validation_drift_decomposition.csv"
HISTORY_FILE = ROOT / "runs" / "expanded_train" / "history.csv"

EARTH_RADIUS = 6378137.0
BASELINE_ATE = 6.914496346411183


# ============================================================
# Generic helpers
# ============================================================

def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def find_column(headers, candidates, required=True):
    for c in candidates:
        if c in headers:
            return c

    if required:
        raise RuntimeError(
            "Could not find any of these columns:\n"
            + "\n".join(candidates)
            + "\n\nAvailable columns:\n"
            + "\n".join(sorted(headers))
        )

    return None


def short_drive(name):
    return (
        name
        .replace("2011_09_26_drive_", "")
        .replace("2011_09_30_drive_", "")
        .replace("2011_10_03_drive_", "")
        .replace("_sync", "")
    )


def save(fig, number, name):
    stem = OUT / f"Figure_{number:02d}_{name}"
    save_publication_figure(fig, stem)
    plt.close(fig)
    print(f"Saved Figure {number:02d}: {name}")


def latlon_to_en(lat, lon, lat0, lon0):
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    north = EARTH_RADIUS * np.radians(lat - lat0)

    mean_lat = np.radians(
        0.5 * (lat + lat0)
    )

    east = (
        EARTH_RADIUS
        * np.cos(mean_lat)
        * np.radians(lon - lon0)
    )

    return east, north


def position_error_m(
    gt_lat,
    gt_lon,
    pred_lat,
    pred_lon,
):
    mean_lat = math.radians(
        0.5 * (gt_lat + pred_lat)
    )

    north = (
        EARTH_RADIUS
        * math.radians(pred_lat - gt_lat)
    )

    east = (
        EARTH_RADIUS
        * math.cos(mean_lat)
        * math.radians(pred_lon - gt_lon)
    )

    return math.hypot(east, north)


def regression_stats(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    slope, intercept = np.polyfit(x, y, 1)

    fitted = slope * x + intercept

    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else float("nan")
    )

    return slope, intercept, r2


# ============================================================
# Load result files
# ============================================================

pred_rows = read_csv(PRED_FILE)
metric_rows = read_csv(METRIC_FILE)
drift_rows = read_csv(DRIFT_FILE)
history = read_csv(HISTORY_FILE)

pred_headers = set(pred_rows[0].keys())
metric_headers = set(metric_rows[0].keys())
drift_headers = set(drift_rows[0].keys())
history_headers = set(history[0].keys())


# ============================================================
# Resolve final-test CSV columns
# ============================================================

drive_col = find_column(
    pred_headers,
    ["drive", "sequence"],
)

gt_lat_t_col = find_column(
    pred_headers,
    ["gt_lat_t"],
)

gt_lon_t_col = find_column(
    pred_headers,
    ["gt_lon_t"],
)

gt_lat_t1_col = find_column(
    pred_headers,
    ["gt_lat_t1"],
)

gt_lon_t1_col = find_column(
    pred_headers,
    ["gt_lon_t1"],
)

pred_lat_t1_col = find_column(
    pred_headers,
    ["pred_lat_t1"],
)

pred_lon_t1_col = find_column(
    pred_headers,
    ["pred_lon_t1"],
)


# ============================================================
# Build trajectories
# ============================================================

drives = sorted({
    row[drive_col]
    for row in pred_rows
})


def build_trajectory(drive):

    rows = [
        r for r in pred_rows
        if r[drive_col] == drive
    ]

    lat0 = float(rows[0][gt_lat_t_col])
    lon0 = float(rows[0][gt_lon_t_col])

    gt_lat = [lat0]
    gt_lon = [lon0]
    pred_lat = [lat0]
    pred_lon = [lon0]

    for row in rows:

        gt_lat.append(
            float(row[gt_lat_t1_col])
        )

        gt_lon.append(
            float(row[gt_lon_t1_col])
        )

        pred_lat.append(
            float(row[pred_lat_t1_col])
        )

        pred_lon.append(
            float(row[pred_lon_t1_col])
        )

    gt_lat = np.asarray(gt_lat)
    gt_lon = np.asarray(gt_lon)
    pred_lat = np.asarray(pred_lat)
    pred_lon = np.asarray(pred_lon)

    gt_e, gt_n = latlon_to_en(
        gt_lat,
        gt_lon,
        lat0,
        lon0,
    )

    pred_e, pred_n = latlon_to_en(
        pred_lat,
        pred_lon,
        lat0,
        lon0,
    )

    return {
        "rows": rows,
        "gt_lat": gt_lat,
        "gt_lon": gt_lon,
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "gt_e": gt_e,
        "gt_n": gt_n,
        "pred_e": pred_e,
        "pred_n": pred_n,
    }


trajectories = {
    d: build_trajectory(d)
    for d in drives
}


# ============================================================
# Figure 1
# Training vs validation local translation RMSE
# ============================================================

epoch_col = find_column(
    history_headers,
    ["epoch"],
)

train_rmse_col = find_column(
    history_headers,
    [
        "train_translation_rmse_m",
        "train_local_rmse_m",
        "train_rmse_m",
    ],
)

val_rmse_col = find_column(
    history_headers,
    [
        "val_translation_rmse_m",
        "validation_translation_rmse_m",
        "val_local_rmse_m",
        "val_rmse_m",
    ],
)

epochs = np.asarray([
    int(float(r[epoch_col]))
    for r in history
])

train_rmse = np.asarray([
    float(r[train_rmse_col])
    for r in history
])

val_rmse = np.asarray([
    float(r[val_rmse_col])
    for r in history
])


fig, ax = plt.subplots(
    figsize=(3.45, 2.55)
)

ax.plot(
    epochs,
    train_rmse,
    color=PRED_COLOR,
    linestyle="-",
    marker="o",
    label="Training",
)

ax.plot(
    epochs,
    val_rmse,
    color=GRAY_DARK,
    linestyle="--",
    marker="s",
    label="Validation",
)

ax.axvline(
    3,
    color=GRAY_MID,
    linestyle=":",
    linewidth=1.0,
)

ax.text(
    3.15,
    ax.get_ylim()[1] * 0.96,
    "Selected epoch",
    fontsize=7,
    va="top",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Local translation RMSE (m)")

subtle_grid(ax)
clean_axis(ax)

ax.legend(
    loc="upper right"
)

fig.tight_layout()

save(
    fig,
    1,
    "Training_vs_Validation_Local_RMSE",
)


# ============================================================
# Figure 2
# Validation step-magnitude RMSE
# ============================================================

step_col = find_column(
    history_headers,
    [
        "val_step_rmse_m",
        "validation_step_rmse_m",
        "val_step_magnitude_rmse_m",
    ],
)

val_step = np.asarray([
    float(r[step_col])
    for r in history
])

fig, ax = plt.subplots(
    figsize=(3.45, 2.55)
)

ax.plot(
    epochs,
    val_step,
    color=PRED_COLOR,
    marker="o",
)

ax.axvline(
    3,
    color=GRAY_MID,
    linestyle=":",
    linewidth=1.0,
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Step-magnitude RMSE (m)")

subtle_grid(ax)
clean_axis(ax)

fig.tight_layout()

save(
    fig,
    2,
    "Validation_Step_Magnitude_RMSE",
)


# ============================================================
# Figure 3
# Validation drift decomposition
# ============================================================

drift_drive_col = find_column(
    drift_headers,
    ["drive", "sequence"],
)

full_col = find_column(
    drift_headers,
    [
        "full_system_ate_rmse_m",
        "A_ate_rmse_m",
        "A_ATE_RMSE_m",
        "actual_system_ate_rmse_m",
    ],
)

heading_col = find_column(
    drift_headers,
    [
        "imu_yaw_only_ate_rmse_m",
        "B_ate_rmse_m",
        "B_ATE_RMSE_m",
        "gt_translation_imu_yaw_ate_rmse_m",
    ],
)

translation_col = find_column(
    drift_headers,
    [
        "translation_only_ate_rmse_m",
        "C_ate_rmse_m",
        "C_ATE_RMSE_m",
        "pred_translation_gt_yaw_ate_rmse_m",
    ],
)


labels = [
    short_drive(r[drift_drive_col])
    for r in drift_rows
]

full_values = [
    float(r[full_col])
    for r in drift_rows
]

heading_values = [
    float(r[heading_col])
    for r in drift_rows
]

translation_values = [
    float(r[translation_col])
    for r in drift_rows
]

x = np.arange(len(labels))
width = 0.24

fig, ax = plt.subplots(
    figsize=(5.9, 3.0)
)

ax.bar(
    x - width,
    full_values,
    width,
    facecolor="white",
    edgecolor=PRED_COLOR,
    linewidth=1.0,
    hatch="//",
    label="Full system",
)

ax.bar(
    x,
    heading_values,
    width,
    facecolor=GRAY_LIGHT,
    edgecolor=PRED_COLOR,
    linewidth=0.8,
    label="GT translation + IMU yaw",
)

ax.bar(
    x + width,
    translation_values,
    width,
    facecolor=GRAY_DARK,
    edgecolor=PRED_COLOR,
    linewidth=0.8,
    label="Predicted translation + GT yaw",
)

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_xlabel("Validation drive")
ax.set_ylabel("Recursive ATE RMSE (m)")

subtle_grid(
    ax,
    axis="y",
)

clean_axis(ax)

ax.legend(
    loc="upper left",
    ncol=1,
)

fig.tight_layout()

save(
    fig,
    3,
    "Validation_Drift_Decomposition",
)


# ============================================================
# Figure 4
# Drift reduction / regularization experiments
# ============================================================

experiments = [
    ("Frozen baseline", 6.914496),
    ("Sequence fine-tuning", 9.9232),
    ("Longitudinal-bias loss", 16.030401),
    ("Gradient aggregation", 9.712741),
    ("Output-layer calibration", 10.596324),
    ("Temporal residual GRU", 9.2392),
    ("Stateful TBPTT GRU", 10.3519),
]

names = [x[0] for x in experiments]
values = [x[1] for x in experiments]

y = np.arange(len(names))

fig, ax = plt.subplots(
    figsize=(5.9, 3.45)
)

bar_faces = [
    PRED_COLOR
    if i == 0
    else GRAY_LIGHT
    for i in range(len(names))
]

bars = ax.barh(
    y,
    values,
    facecolor=bar_faces,
    edgecolor=PRED_COLOR,
    linewidth=0.7,
)

ax.set_yticks(y)
ax.set_yticklabels(names)

ax.invert_yaxis()

ax.axvline(
    BASELINE_ATE,
    color=GRAY_DARK,
    linestyle="--",
    linewidth=0.9,
)

for bar, value in zip(bars, values):

    ax.text(
        value + 0.15,
        bar.get_y() + bar.get_height()/2,
        f"{value:.2f}",
        va="center",
        fontsize=7,
    )

ax.set_xlabel(
    "Validation recursive ATE RMSE (m)"
)

subtle_grid(
    ax,
    axis="x",
)

clean_axis(ax)

fig.tight_layout()

save(
    fig,
    4,
    "Drift_Reduction_Experiment_Comparison",
)


# ============================================================
# Metrics file columns
# ============================================================

metric_drive_col = find_column(
    metric_headers,
    ["drive", "sequence"],
)

metric_ate_col = find_column(
    metric_headers,
    [
        "recursive_ate_rmse_m",
        "ate_rmse_m",
    ],
)

metric_final_col = find_column(
    metric_headers,
    [
        "final_position_error_m",
        "final_error_m",
    ],
    required=False,
)

metrics_by_drive = {
    r[metric_drive_col]: r
    for r in metric_rows
}


# ============================================================
# Figures 5–7
# Per-sequence evaluation suites
# ============================================================

figure_number = 5


for drive in drives:

    t = trajectories[drive]
    short = short_drive(drive)

    errors = np.asarray([
        position_error_m(
            float(glat),
            float(glon),
            float(plat),
            float(plon),
        )
        for glat, glon, plat, plon
        in zip(
            t["gt_lat"][1:],
            t["gt_lon"][1:],
            t["pred_lat"][1:],
            t["pred_lon"][1:],
        )
    ])


    fig = plt.figure(
        figsize=(6.8, 6.9)
    )


    # --------------------------------------------------------
    # Main trajectory
    # --------------------------------------------------------

    ax = fig.add_axes([
        0.11,
        0.43,
        0.82,
        0.47,
    ])

    ax.plot(
        t["gt_e"],
        t["gt_n"],
        color=GT_COLOR,
        linewidth=1.5,
        label="Ground Truth",
    )

    ax.plot(
        t["pred_e"],
        t["pred_n"],
        color=PRED_COLOR,
        linewidth=1.3,
        label="Model Prediction",
    )

    # Start marker
    ax.plot(
        t["gt_e"][0],
        t["gt_n"][0],
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgecolor=PRED_COLOR,
        linestyle="None",
    )

    # Ground-truth end
    ax.plot(
        t["gt_e"][-1],
        t["gt_n"][-1],
        marker="x",
        markersize=6,
        color=GT_COLOR,
        linestyle="None",
    )

    # Predicted end
    ax.plot(
        t["pred_e"][-1],
        t["pred_n"][-1],
        marker="x",
        markersize=6,
        color=PRED_COLOR,
        linestyle="None",
    )

    ax.set_xlabel(
        "East displacement (m)"
    )

    ax.set_ylabel(
        "North displacement (m)"
    )

    ax.set_aspect(
        "equal",
        adjustable="datalim",
    )

    subtle_grid(ax)
    clean_axis(ax)

    # Required top-right legend, outside data region
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.12),
        ncol=2,
    )


    # --------------------------------------------------------
    # Zoom panel bottom-left
    # --------------------------------------------------------

    n = len(t["gt_e"])

    center = int(0.62 * n)

    half = max(
        18,
        min(
            65,
            n // 12,
        ),
    )

    start = max(
        1,
        center - half,
    )

    end = min(
        n - 1,
        center + half,
    )


    zoom = fig.add_axes([
        0.10,
        0.07,
        0.36,
        0.22,
    ])

    zoom.plot(
        t["gt_e"][start:end],
        t["gt_n"][start:end],
        color=GT_COLOR,
        linewidth=1.2,
    )

    zoom.plot(
        t["pred_e"][start:end],
        t["pred_n"][start:end],
        color=PRED_COLOR,
        linewidth=1.1,
    )

    zoom.set_xlabel(
        "East (m)",
        fontsize=8,
    )

    zoom.set_ylabel(
        "North (m)",
        fontsize=8,
    )

    zoom.tick_params(
        labelsize=7,
    )

    subtle_grid(zoom)
    clean_axis(zoom)

    zoom.set_title(
        "(a) Zoomed section",
        loc="left",
        fontsize=8,
        pad=5,
    )


    # --------------------------------------------------------
    # ATE panel bottom-right
    # --------------------------------------------------------

    ate_ax = fig.add_axes([
        0.57,
        0.07,
        0.36,
        0.22,
    ])

    frame_index = np.arange(
        1,
        len(errors) + 1,
    )

    ate_ax.plot(
        frame_index,
        errors,
        color=PRED_COLOR,
        linewidth=1.0,
    )

    rolling_window = min(
        50,
        max(
            10,
            len(errors)//20,
        ),
    )

    if len(errors) >= rolling_window:

        rolling = np.convolve(
            errors,
            np.ones(rolling_window)
            / rolling_window,
            mode="valid",
        )

        ate_ax.plot(
            frame_index[
                rolling_window - 1:
            ],
            rolling,
            color=GRAY_MID,
            linestyle="--",
            linewidth=1.0,
        )

    ate_ax.set_xlabel(
        "Frame index",
        fontsize=8,
    )

    ate_ax.set_ylabel(
        "ATE (m)",
        fontsize=8,
    )

    ate_ax.tick_params(
        labelsize=7,
    )

    subtle_grid(ate_ax)
    clean_axis(ate_ax)

    ate_ax.set_title(
        "(b) ATE evolution",
        loc="left",
        fontsize=8,
        pad=5,
    )


    fig.text(
        0.02,
        0.965,
        f"Sequence {short}",
        fontsize=9,
        fontweight="bold",
    )


    save(
        fig,
        figure_number,
        f"Sequence_{short}_Evaluation_Suite",
    )

    figure_number += 1


# ============================================================
# Figure 8
# Final-test ATE by sequence
# ============================================================

ate_labels = []
ate_values = []

for row in metric_rows:

    ate_labels.append(
        short_drive(
            row[metric_drive_col]
        )
    )

    ate_values.append(
        float(
            row[metric_ate_col]
        )
    )


fig, ax = plt.subplots(
    figsize=(3.45, 2.55)
)

bars = ax.bar(
    ate_labels,
    ate_values,
    facecolor="white",
    edgecolor=PRED_COLOR,
    linewidth=1.0,
)

for bar, value in zip(
    bars,
    ate_values,
):
    ax.text(
        bar.get_x()
        + bar.get_width()/2,
        value + max(ate_values)*0.025,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=7,
    )

ax.set_xlabel("Final-test sequence")
ax.set_ylabel("Recursive ATE RMSE (m)")

subtle_grid(
    ax,
    axis="y",
)

clean_axis(ax)

fig.tight_layout()

save(
    fig,
    8,
    "Final_Test_ATE_By_Sequence",
)


# ============================================================
# Figure 9
# Local translation error histogram
# ============================================================

vector_sets = [
    (
        "pred_dx",
        "pred_dy",
        "target_dx",
        "target_dy",
    ),
    (
        "pred_dx_m",
        "pred_dy_m",
        "gt_dx_m",
        "gt_dy_m",
    ),
    (
        "predicted_dx",
        "predicted_dy",
        "target_dx",
        "target_dy",
    ),
]

local_errors = None


for px, py, tx, ty in vector_sets:

    if {
        px,
        py,
        tx,
        ty,
    }.issubset(pred_headers):

        local_errors = np.asarray([
            math.hypot(
                float(r[px])
                - float(r[tx]),
                float(r[py])
                - float(r[ty]),
            )
            for r in pred_rows
        ])

        break


if local_errors is None:

    scalar_col = find_column(
        pred_headers,
        [
            "local_translation_error_m",
            "translation_error_m",
            "local_error_m",
        ],
    )

    local_errors = np.asarray([
        float(r[scalar_col])
        for r in pred_rows
    ])


local_rmse = math.sqrt(
    float(
        np.mean(
            local_errors ** 2
        )
    )
)

local_median = float(
    np.median(
        local_errors
    )
)


fig, ax = plt.subplots(
    figsize=(3.45, 2.55)
)

ax.hist(
    local_errors,
    bins=50,
    facecolor=GRAY_LIGHT,
    edgecolor=PRED_COLOR,
    linewidth=0.45,
)

ax.axvline(
    local_rmse,
    color=PRED_COLOR,
    linestyle="--",
    linewidth=1.0,
    label=f"RMSE = {local_rmse:.3f} m",
)

ax.axvline(
    local_median,
    color=GRAY_DARK,
    linestyle=":",
    linewidth=1.0,
    label=f"Median = {local_median:.3f} m",
)

ax.set_xlabel(
    "Frame-to-frame translation error (m)"
)

ax.set_ylabel(
    "Frame-pair count"
)

clean_axis(ax)

ax.legend(
    loc="upper right"
)

fig.tight_layout()

save(
    fig,
    9,
    "Final_Test_Local_Translation_Error_Histogram",
)


# ============================================================
# Figure 10
# KITTI-style 2D relative translation error
# ============================================================

segment_lengths = [
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
]

valid_lengths = []
mean_errors = []
median_errors = []


for requested_length in segment_lengths:

    values = []

    for drive in drives:

        t = trajectories[drive]

        gt_xy = np.column_stack([
            t["gt_e"],
            t["gt_n"],
        ])

        pred_xy = np.column_stack([
            t["pred_e"],
            t["pred_n"],
        ])

        gt_steps = np.linalg.norm(
            np.diff(
                gt_xy,
                axis=0,
            ),
            axis=1,
        )

        cumulative = np.concatenate([
            [0.0],
            np.cumsum(gt_steps),
        ])

        for start in range(
            len(cumulative) - 1
        ):

            target = (
                cumulative[start]
                + requested_length
            )

            if target > cumulative[-1]:
                break

            end = int(
                np.searchsorted(
                    cumulative,
                    target,
                    side="left",
                )
            )

            if end <= start:
                continue

            actual_length = (
                cumulative[end]
                - cumulative[start]
            )

            if actual_length <= 0:
                continue

            gt_delta = (
                gt_xy[end]
                - gt_xy[start]
            )

            pred_delta = (
                pred_xy[end]
                - pred_xy[start]
            )

            error = np.linalg.norm(
                pred_delta
                - gt_delta
            )

            values.append(
                100.0
                * error
                / actual_length
            )

    if values:

        valid_lengths.append(
            requested_length
        )

        mean_errors.append(
            float(np.mean(values))
        )

        median_errors.append(
            float(np.median(values))
        )


fig, ax = plt.subplots(
    figsize=(3.45, 2.55)
)

ax.plot(
    valid_lengths,
    mean_errors,
    color=PRED_COLOR,
    marker="o",
    label="Mean",
)

ax.plot(
    valid_lengths,
    median_errors,
    color=GRAY_DARK,
    linestyle="--",
    marker="s",
    label="Median",
)

ax.set_xlabel(
    "Ground-truth segment length (m)"
)

ax.set_ylabel(
    "Relative translation error (%)"
)

subtle_grid(ax)
clean_axis(ax)

ax.legend(
    loc="upper right"
)

fig.tight_layout()

save(
    fig,
    10,
    "KITTI_Style_2D_Error_vs_Path_Length",
)


# ============================================================
# Longest final-test drive for regression figures
# ============================================================

longest_drive = max(
    drives,
    key=lambda d:
        len(
            trajectories[d]["rows"]
        ),
)

t = trajectories[
    longest_drive
]

short = short_drive(
    longest_drive
)


# ============================================================
# Figure 11
# Longitude agreement
# ============================================================

slope, intercept, r2 = regression_stats(
    t["gt_lon"],
    t["pred_lon"],
)

minimum = min(
    t["gt_lon"].min(),
    t["pred_lon"].min(),
)

maximum = max(
    t["gt_lon"].max(),
    t["pred_lon"].max(),
)

line_x = np.linspace(
    minimum,
    maximum,
    300,
)


fig, ax = plt.subplots(
    figsize=(3.3, 3.3)
)

ax.scatter(
    t["gt_lon"],
    t["pred_lon"],
    s=5,
    facecolor=PRED_COLOR,
    edgecolor="none",
    alpha=0.55,
    label="Prediction",
)

ax.plot(
    line_x,
    line_x,
    color=GT_COLOR,
    linewidth=1.2,
    label="Ideal agreement",
)

ax.plot(
    line_x,
    slope*line_x + intercept,
    color=GRAY_MID,
    linestyle="--",
    linewidth=1.0,
    label="Linear fit",
)

ax.set_xlabel(
    "Ground-truth longitude (deg)"
)

ax.set_ylabel(
    "Predicted longitude (deg)"
)

ax.ticklabel_format(
    style="plain",
    axis="both",
    useOffset=False,
)

clean_axis(ax)

ax.legend(
    loc="lower right"
)

ax.text(
    0.04,
    0.96,
    (
        f"Slope = {slope:.5f}\n"
        f"$R^2$ = {r2:.5f}"
    ),
    transform=ax.transAxes,
    va="top",
    fontsize=7,
)

fig.tight_layout()

save(
    fig,
    11,
    f"Longitude_Agreement_Sequence_{short}",
)


# ============================================================
# Figure 12
# Latitude agreement
# ============================================================

slope, intercept, r2 = regression_stats(
    t["gt_lat"],
    t["pred_lat"],
)

minimum = min(
    t["gt_lat"].min(),
    t["pred_lat"].min(),
)

maximum = max(
    t["gt_lat"].max(),
    t["pred_lat"].max(),
)

line_x = np.linspace(
    minimum,
    maximum,
    300,
)


fig, ax = plt.subplots(
    figsize=(3.3, 3.3)
)

ax.scatter(
    t["gt_lat"],
    t["pred_lat"],
    s=5,
    facecolor=PRED_COLOR,
    edgecolor="none",
    alpha=0.55,
    label="Prediction",
)

ax.plot(
    line_x,
    line_x,
    color=GT_COLOR,
    linewidth=1.2,
    label="Ideal agreement",
)

ax.plot(
    line_x,
    slope*line_x + intercept,
    color=GRAY_MID,
    linestyle="--",
    linewidth=1.0,
    label="Linear fit",
)

ax.set_xlabel(
    "Ground-truth latitude (deg)"
)

ax.set_ylabel(
    "Predicted latitude (deg)"
)

ax.ticklabel_format(
    style="plain",
    axis="both",
    useOffset=False,
)

clean_axis(ax)

ax.legend(
    loc="lower right"
)

ax.text(
    0.04,
    0.96,
    (
        f"Slope = {slope:.5f}\n"
        f"$R^2$ = {r2:.5f}"
    ),
    transform=ax.transAxes,
    va="top",
    fontsize=7,
)

fig.tight_layout()

save(
    fig,
    12,
    f"Latitude_Agreement_Sequence_{short}",
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 78)
print("PUBLICATION THESIS FIGURE PACKAGE COMPLETE")
print("=" * 78)

print("Figures              : 12")
print("Ground Truth         : blue")
print("Model Prediction     : black")
print("Other information    : grayscale")
print("Trajectory legend    : top-right")
print("Zoom panel           : bottom-left, outside main trajectory")
print("ATE panel            : bottom-right, outside main trajectory")
print("Raster resolution    : 600 dpi")
print("Vector output        : PDF + SVG")
print("Model rerun          : NO")
print("Predictions modified : NO")

print()
print("Output:")
print(OUT)
print("=" * 78)
