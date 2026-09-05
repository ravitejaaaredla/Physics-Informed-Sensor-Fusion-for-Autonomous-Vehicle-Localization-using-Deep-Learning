from pathlib import Path
import csv
import json

ROOT = Path.cwd()

OUT = (
    ROOT
    / "results"
    / "thesis_package"
    / "thesis_results_tables.md"
)

PERFORMANCE = (
    ROOT
    / "results"
    / "thesis_package"
    / "final_performance_table.csv"
)

DRIFT = (
    ROOT
    / "results"
    / "validation_drift_decomposition.csv"
)

SPLIT = (
    ROOT
    / "results"
    / "new_drive_level_split.csv"
)

FREEZE = (
    ROOT
    / "results"
    / "final_frozen_checkpoint.json"
)


with PERFORMANCE.open(
    newline="",
    encoding="utf-8",
) as f:
    performance = list(
        csv.DictReader(f)
    )


with DRIFT.open(
    newline="",
    encoding="utf-8",
) as f:
    drift = list(
        csv.DictReader(f)
    )


freeze = json.loads(
    FREEZE.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# Helpers
# ============================================================

def short_drive(name):

    return (
        name
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


def f3(value):

    if value in (
        "",
        None,
    ):
        return "—"

    return f"{float(value):.3f}"


def f2(value):

    if value in (
        "",
        None,
    ):
        return "—"

    return f"{float(value):.2f}"


# ============================================================
# Build markdown
# ============================================================

lines = []

lines.append(
    "# Thesis Results Tables"
)

lines.append("")

lines.append(
    "All reported results correspond to the frozen canonical "
    "SensorFusionMotionModel checkpoint selected using validation "
    "recursive ATE RMSE."
)

lines.append("")


# ============================================================
# Table 1 — Canonical model
# ============================================================

lines.append(
    "## Table 1. Canonical Model Configuration"
)

lines.append("")

lines.append(
    "| Item | Value |"
)

lines.append(
    "|---|---|"
)

lines.append(
    "| Canonical model | SensorFusionMotionModel |"
)

lines.append(
    "| Model source | `src/models/sensor_fusion.py` |"
)

lines.append(
    "| Model parameters | 785,346 |"
)

lines.append(
    "| Neural output | Local relative translation `[dx, dy]` |"
)

lines.append(
    "| Camera input | `camera_t`, `camera_t1` |"
)

lines.append(
    "| LiDAR input | `lidar_t`, `lidar_t1` |"
)

lines.append(
    "| IMU input | 5 x 6 IMU window |"
)

lines.append(
    "| Heading propagation | Raw IMU `wz` trapezoidal integration |"
)

lines.append(
    "| Initial global state | Latitude, longitude and yaw once per contiguous sequence |"
)

lines.append(
    "| Selected epoch | "
    f"{freeze['selected_epoch']} |"
)

lines.append(
    "| Validation selection metric | Recursive ATE RMSE |"
)

lines.append(
    "| Frozen checkpoint SHA256 | "
    f"`{freeze['sha256']}` |"
)

lines.append("")


# ============================================================
# Table 2 — Data split
# ============================================================

lines.append(
    "## Table 2. Dataset Split"
)

lines.append("")

lines.append(
    "| Split | Drives | Valid frame pairs | Role |"
)

lines.append(
    "|---|---:|---:|---|"
)

lines.append(
    "| Training | 16 | 19,483 | Gradient-based training |"
)

lines.append(
    "| Validation | 3 | 2,121 | Checkpoint selection and diagnostics |"
)

lines.append(
    "| Diagnostic Drive 0018 | 1 | 269 | Historical diagnostic only |"
)

lines.append(
    "| Final test | 3 | 2,197 | Final frozen-model evaluation |"
)

lines.append(
    "| **Total** | **23** | **24,070** | — |"
)

lines.append("")


# ============================================================
# Table 3 — Final-test results
# ============================================================

lines.append(
    "## Table 3. Final-Test Performance"
)

lines.append("")

lines.append(
    "| Drive | Pairs | Local RMSE (m) | "
    "GT distance (m) | Pred. distance (m) | "
    "Pred/GT | Recursive ATE (m) | Final error (m) |"
)

lines.append(
    "|---|---:|---:|---:|---:|---:|---:|---:|"
)


for row in performance:

    if row[
        "evaluation_scope"
    ] != "final_test_drive":
        continue

    lines.append(
        "| "
        f"{short_drive(row['drive'])} | "
        f"{row['pairs']} | "
        f"{f3(row['local_translation_rmse_m'])} | "
        f"{f2(row['gt_distance_m'])} | "
        f"{f2(row['pred_distance_m'])} | "
        f"{f3(row['pred_gt_distance_ratio'])} | "
        f"{f3(row['recursive_ate_rmse_m'])} | "
        f"{f3(row['final_position_error_m'])} |"
    )


aggregate = next(
    row
    for row in performance
    if row[
        "evaluation_scope"
    ] == "final_test_aggregate"
)


lines.append(
    "| **Aggregate** | "
    f"**{aggregate['pairs']}** | "
    f"**{f3(aggregate['local_translation_rmse_m'])}** | "
    f"**{f2(aggregate['gt_distance_m'])}** | "
    f"**{f2(aggregate['pred_distance_m'])}** | "
    f"**{f3(aggregate['pred_gt_distance_ratio'])}** | "
    f"**{f3(aggregate['recursive_ate_rmse_m'])}** | "
    "— |"
)

lines.append("")


# ============================================================
# Table 4 — Validation drift decomposition
# ============================================================

lines.append(
    "## Table 4. Validation Drift Decomposition"
)

lines.append("")

lines.append(
    "| Drive | GT distance (m) | Full system ATE (m) | "
    "GT translation + IMU yaw ATE (m) | "
    "Predicted translation + GT yaw ATE (m) | "
    "IMU yaw MAE (deg) |"
)

lines.append(
    "|---|---:|---:|---:|---:|---:|"
)


for row in drift:

    lines.append(
        "| "
        f"{short_drive(row['drive'])} | "
        f"{f2(row['gt_distance_m'])} | "
        f"{f3(row['full_system_ate_rmse_m'])} | "
        f"{f3(row['imu_yaw_only_ate_rmse_m'])} | "
        f"{f3(row['translation_only_ate_rmse_m'])} | "
        f"{f3(row['imu_yaw_mae_deg'])} |"
    )


lines.append("")


# ============================================================
# Table 5 — Drift-improvement experiments
# ============================================================

lines.append(
    "## Table 5. Drift-Reduction Experiments"
)

lines.append("")

lines.append(
    "| Experiment | Validation recursive ATE (m) | Decision |"
)

lines.append(
    "|---|---:|---|"
)

lines.append(
    "| Frozen canonical checkpoint | **6.914** | **Retained** |"
)

lines.append(
    "| Sequence fine-tuning with frozen BatchNorm | 9.923 | Rejected |"
)

lines.append(
    "| Longitudinal-bias full-network probe | 16.030 | Rejected |"
)

lines.append(
    "| Multi-window gradient aggregation | 9.713 | Rejected |"
)

lines.append(
    "| Output-layer-only calibration | 10.596 | Rejected |"
)

lines.append(
    "| Temporal residual GRU | 9.239 | Rejected |"
)

lines.append(
    "| Stateful temporal residual GRU with TBPTT | 10.352 | Rejected |"
)

lines.append("")


# ============================================================
# Table 6 — Key final metrics
# ============================================================

validation = next(
    row
    for row in performance
    if row[
        "evaluation_scope"
    ] == "validation_aggregate"
)


lines.append(
    "## Table 6. Key Aggregate Metrics"
)

lines.append("")

lines.append(
    "| Metric | Validation | Final test |"
)

lines.append(
    "|---|---:|---:|"
)

lines.append(
    "| Pairs | "
    f"{validation['pairs']} | "
    f"{aggregate['pairs']} |"
)

lines.append(
    "| Local translation RMSE (m) | "
    f"{f3(validation['local_translation_rmse_m'])} | "
    f"{f3(aggregate['local_translation_rmse_m'])} |"
)

lines.append(
    "| Step RMSE (m) | "
    f"{f3(validation['step_rmse_m'])} | "
    f"{f3(aggregate['step_rmse_m'])} |"
)

lines.append(
    "| Recursive ATE RMSE (m) | "
    f"{f3(validation['recursive_ate_rmse_m'])} | "
    f"{f3(aggregate['recursive_ate_rmse_m'])} |"
)

lines.append(
    "| Recursive ATE mean (m) | "
    f"{f3(validation['recursive_ate_mean_m'])} | "
    f"{f3(aggregate['recursive_ate_mean_m'])} |"
)

lines.append(
    "| Recursive ATE median (m) | "
    f"{f3(validation['recursive_ate_median_m'])} | "
    f"{f3(aggregate['recursive_ate_median_m'])} |"
)

lines.append(
    "| Recursive ATE maximum (m) | "
    f"{f3(validation['recursive_ate_max_m'])} | "
    f"{f3(aggregate['recursive_ate_max_m'])} |"
)

lines.append("")


# ============================================================
# Interpretation notes
# ============================================================

lines.append(
    "## Thesis Interpretation Notes"
)

lines.append("")

lines.append(
    "- Translation error, particularly systematic longitudinal "
    "`dx` bias, was the dominant source of recursive drift."
)

lines.append(
    "- IMU yaw integration contributed substantially less error "
    "than learned translation on the validation drives."
)

lines.append(
    "- The direction of longitudinal bias varied across drives, "
    "preventing a reliable single global scale correction."
)

lines.append(
    "- Multiple drift-reduction approaches improved or preserved "
    "local RMSE but degraded recursive cross-drive ATE."
)

lines.append(
    "- Therefore the originally selected frozen checkpoint was "
    "retained as the canonical final model."
)

lines.append("")

lines.append(
    "- The final-test aggregate travelled-distance ratio was "
    f"{float(aggregate['pred_gt_distance_ratio']):.3f}, "
    "showing good overall motion-scale calibration despite "
    "drive-dependent accumulated trajectory drift."
)

lines.append("")


OUT.write_text(
    "\n".join(lines),
    encoding="utf-8"
)


print()
print("=" * 100)
print("THESIS RESULTS TABLE PACKAGE")
print("=" * 100)

print(
    "Tables created : 6"
)

print(
    "Output         :",
    OUT
)

print()
print(
    "Contains:"
)

print(
    "  1. Canonical model configuration"
)

print(
    "  2. Dataset split"
)

print(
    "  3. Final-test performance"
)

print(
    "  4. Validation drift decomposition"
)

print(
    "  5. Drift-reduction experiments"
)

print(
    "  6. Key aggregate metrics"
)

print()

print(
    "PASS: thesis-ready result tables created."
)

print("=" * 100)
