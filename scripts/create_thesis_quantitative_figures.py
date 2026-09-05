from pathlib import Path
import csv

import matplotlib.pyplot as plt


ROOT = Path.cwd()

OUT = (
    ROOT
    / "results"
    / "thesis_package"
    / "figures"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 1. TRAINING / VALIDATION LOCAL RMSE
# ============================================================

history_file = (
    ROOT
    / "runs"
    / "expanded_train"
    / "history.csv"
)

with history_file.open(
    newline="",
    encoding="utf-8",
) as f:
    history = list(
        csv.DictReader(f)
    )


epochs = [
    int(r["epoch"])
    for r in history
]

train_rmse = [
    float(
        r[
            "train_translation_rmse_m"
        ]
    )
    for r in history
]

val_rmse = [
    float(
        r[
            "val_translation_rmse_m"
        ]
    )
    for r in history
]


fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)

ax.plot(
    epochs,
    train_rmse,
    marker="o",
    label="Training local RMSE",
)

ax.plot(
    epochs,
    val_rmse,
    marker="o",
    label="Validation local RMSE",
)

ax.axvline(
    3,
    linestyle="--",
    label="Selected checkpoint (epoch 3)",
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Translation RMSE (m)"
)

ax.set_title(
    "Local Translation Error During Training"
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

path1 = (
    OUT
    / "training_validation_local_rmse.png"
)

fig.savefig(
    path1,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 2. VALIDATION RECURSIVE ATE THROUGH TRAINING
# ============================================================

val_ate = [
    float(
        r[
            "val_recursive_ate_rmse_m"
        ]
    )
    for r in history
]


fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)

ax.plot(
    epochs,
    val_ate,
    marker="o",
)

ax.scatter(
    [3],
    [val_ate[2]],
    s=90,
    label=(
        f"Selected epoch 3: "
        f"{val_ate[2]:.3f} m"
    ),
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Recursive ATE RMSE (m)"
)

ax.set_title(
    "Validation Recursive Localization Error"
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

path2 = (
    OUT
    / "validation_recursive_ate_training_curve.png"
)

fig.savefig(
    path2,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 3. DRIFT DECOMPOSITION
# ============================================================

drift_file = (
    ROOT
    / "results"
    / "validation_drift_decomposition.csv"
)

with drift_file.open(
    newline="",
    encoding="utf-8",
) as f:
    drift = list(
        csv.DictReader(f)
    )


labels = []

full_system = []
imu_only = []
translation_only = []


for row in drift:

    drive = (
        row["drive"]
        .replace(
            "2011_09_26_drive_",
            ""
        )
        .replace(
            "2011_09_30_drive_",
            ""
        )
        .replace(
            "_sync",
            ""
        )
    )

    labels.append(
        f"Drive {drive}"
    )

    full_system.append(
        float(
            row[
                "full_system_ate_rmse_m"
            ]
        )
    )

    imu_only.append(
        float(
            row[
                "imu_yaw_only_ate_rmse_m"
            ]
        )
    )

    translation_only.append(
        float(
            row[
                "translation_only_ate_rmse_m"
            ]
        )
    )


x = list(
    range(
        len(labels)
    )
)

width = 0.25


fig, ax = plt.subplots(
    figsize=(9.0, 5.8)
)

ax.bar(
    [
        i - width
        for i in x
    ],
    full_system,
    width=width,
    label="Full system",
)

ax.bar(
    x,
    imu_only,
    width=width,
    label="GT translation + IMU yaw",
)

ax.bar(
    [
        i + width
        for i in x
    ],
    translation_only,
    width=width,
    label="Predicted translation + GT yaw",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    labels
)

ax.set_ylabel(
    "Recursive ATE RMSE (m)"
)

ax.set_title(
    "Validation Drift Decomposition"
)

ax.grid(
    True,
    axis="y",
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

path3 = (
    OUT
    / "validation_drift_decomposition.png"
)

fig.savefig(
    path3,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 4. FINAL-TEST PER-DRIVE ATE
# ============================================================

final_file = (
    ROOT
    / "results"
    / "final_test_drive_metrics.csv"
)

with final_file.open(
    newline="",
    encoding="utf-8",
) as f:
    final_rows = list(
        csv.DictReader(f)
    )


final_labels = []
ates = []
final_errors = []


for row in final_rows:

    drive = (
        row["drive"]
        .replace(
            "2011_09_26_drive_",
            ""
        )
        .replace(
            "2011_09_30_drive_",
            ""
        )
        .replace(
            "_sync",
            ""
        )
    )

    final_labels.append(
        f"Drive {drive}"
    )

    ates.append(
        float(
            row[
                "recursive_ate_rmse_m"
            ]
        )
    )

    final_errors.append(
        float(
            row[
                "final_position_error_m"
            ]
        )
    )


x = list(
    range(
        len(final_labels)
    )
)

width = 0.34


fig, ax = plt.subplots(
    figsize=(8.5, 5.8)
)

ax.bar(
    [
        i - width / 2
        for i in x
    ],
    ates,
    width=width,
    label="ATE RMSE",
)

ax.bar(
    [
        i + width / 2
        for i in x
    ],
    final_errors,
    width=width,
    label="Final position error",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    final_labels
)

ax.set_ylabel(
    "Position error (m)"
)

ax.set_title(
    "Final-Test Recursive Localization Performance"
)

ax.grid(
    True,
    axis="y",
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

path4 = (
    OUT
    / "final_test_per_drive_error.png"
)

fig.savefig(
    path4,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("=" * 96)
print("THESIS QUANTITATIVE FIGURES")
print("=" * 96)

print(
    "Saved:",
    path1.name
)

print(
    "Saved:",
    path2.name
)

print(
    "Saved:",
    path3.name
)

print(
    "Saved:",
    path4.name
)

print()
print(
    "Output folder:",
    OUT
)

print()
print(
    "PASS: thesis quantitative figures created "
    "from locked canonical results."
)

print("=" * 96)
