from pathlib import Path
import csv
import json

ROOT = Path.cwd()

FREEZE = (
    ROOT
    / "results"
    / "final_frozen_checkpoint.json"
)

HISTORY = (
    ROOT
    / "runs"
    / "expanded_train"
    / "history.csv"
)

FINAL_SUMMARY = (
    ROOT
    / "results"
    / "final_test_summary.json"
)

FINAL_DRIVES = (
    ROOT
    / "results"
    / "final_test_drive_metrics.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "thesis_package"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_CSV = (
    OUT_DIR
    / "final_performance_table.csv"
)

OUT_JSON = (
    OUT_DIR
    / "final_performance_summary.json"
)


# ============================================================
# Load canonical records
# ============================================================

freeze = json.loads(
    FREEZE.read_text(
        encoding="utf-8"
    )
)

final_summary = json.loads(
    FINAL_SUMMARY.read_text(
        encoding="utf-8"
    )
)


with HISTORY.open(
    newline="",
    encoding="utf-8",
) as f:

    history = list(
        csv.DictReader(f)
    )


selected_epoch = int(
    freeze["selected_epoch"]
)


selected_history = None

for row in history:

    if int(row["epoch"]) == selected_epoch:

        selected_history = row
        break


if selected_history is None:

    raise RuntimeError(
        f"Selected epoch {selected_epoch} "
        "not found in history.csv"
    )


with FINAL_DRIVES.open(
    newline="",
    encoding="utf-8",
) as f:

    final_drive_rows = list(
        csv.DictReader(f)
    )


# ============================================================
# Build canonical performance table
# ============================================================

fields = [
    "evaluation_scope",
    "drive",
    "pairs",

    "local_translation_rmse_m",
    "step_rmse_m",

    "recursive_ate_rmse_m",
    "recursive_ate_mean_m",
    "recursive_ate_median_m",
    "recursive_ate_max_m",

    "gt_distance_m",
    "pred_distance_m",
    "pred_gt_distance_ratio",

    "final_position_error_m",
]


rows = []


# ------------------------------------------------------------
# Validation aggregate
# ------------------------------------------------------------

rows.append({
    "evaluation_scope":
        "validation_aggregate",

    "drive":
        "3 unseen validation drives",

    "pairs":
        2121,

    "local_translation_rmse_m":
        float(
            selected_history[
                "val_translation_rmse_m"
            ]
        ),

    "step_rmse_m":
        float(
            selected_history[
                "val_step_rmse_m"
            ]
        ),

    "recursive_ate_rmse_m":
        float(
            selected_history[
                "val_recursive_ate_rmse_m"
            ]
        ),

    "recursive_ate_mean_m":
        float(
            selected_history[
                "val_recursive_ate_mean_m"
            ]
        ),

    "recursive_ate_median_m":
        float(
            selected_history[
                "val_recursive_ate_median_m"
            ]
        ),

    "recursive_ate_max_m":
        float(
            selected_history[
                "val_recursive_ate_max_m"
            ]
        ),

    "gt_distance_m":
        "",

    "pred_distance_m":
        "",

    "pred_gt_distance_ratio":
        "",

    "final_position_error_m":
        "",
})


# ------------------------------------------------------------
# Final-test aggregate
# ------------------------------------------------------------

rows.append({
    "evaluation_scope":
        "final_test_aggregate",

    "drive":
        "3 final-test drives",

    "pairs":
        int(
            final_summary[
                "final_test_pairs"
            ]
        ),

    "local_translation_rmse_m":
        float(
            final_summary[
                "local_translation_rmse_m"
            ]
        ),

    "step_rmse_m":
        float(
            final_summary[
                "step_rmse_m"
            ]
        ),

    "recursive_ate_rmse_m":
        float(
            final_summary[
                "recursive_ate_rmse_m"
            ]
        ),

    "recursive_ate_mean_m":
        float(
            final_summary[
                "recursive_ate_mean_m"
            ]
        ),

    "recursive_ate_median_m":
        float(
            final_summary[
                "recursive_ate_median_m"
            ]
        ),

    "recursive_ate_max_m":
        float(
            final_summary[
                "recursive_ate_max_m"
            ]
        ),

    "gt_distance_m":
        float(
            final_summary[
                "total_gt_distance_m"
            ]
        ),

    "pred_distance_m":
        float(
            final_summary[
                "total_pred_distance_m"
            ]
        ),

    "pred_gt_distance_ratio":
        float(
            final_summary[
                "overall_pred_gt_distance_ratio"
            ]
        ),

    "final_position_error_m":
        "",
})


# ------------------------------------------------------------
# Final-test individual drives
# ------------------------------------------------------------

for r in final_drive_rows:

    rows.append({
        "evaluation_scope":
            "final_test_drive",

        "drive":
            r["drive"],

        "pairs":
            int(
                r["pairs"]
            ),

        "local_translation_rmse_m":
            float(
                r[
                    "local_translation_rmse_m"
                ]
            ),

        "step_rmse_m":
            float(
                r[
                    "step_rmse_m"
                ]
            ),

        "recursive_ate_rmse_m":
            float(
                r[
                    "recursive_ate_rmse_m"
                ]
            ),

        "recursive_ate_mean_m":
            float(
                r[
                    "recursive_ate_mean_m"
                ]
            ),

        "recursive_ate_median_m":
            float(
                r[
                    "recursive_ate_median_m"
                ]
            ),

        "recursive_ate_max_m":
            float(
                r[
                    "recursive_ate_max_m"
                ]
            ),

        "gt_distance_m":
            float(
                r[
                    "gt_distance_m"
                ]
            ),

        "pred_distance_m":
            float(
                r[
                    "pred_distance_m"
                ]
            ),

        "pred_gt_distance_ratio":
            float(
                r[
                    "pred_gt_distance_ratio"
                ]
            ),

        "final_position_error_m":
            float(
                r[
                    "final_position_error_m"
                ]
            ),
    })


with OUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# Compact thesis summary
# ============================================================

summary = {
    "canonical_model":
        "SensorFusionMotionModel",

    "model_source":
        "src/models/sensor_fusion.py",

    "checkpoint":
        "runs/expanded_train/best_model.pth",

    "checkpoint_sha256":
        freeze["sha256"],

    "selected_epoch":
        selected_epoch,

    "checkpoint_selection":
        "validation recursive ATE RMSE",

    "validation": {
        "pairs":
            2121,

        "recursive_ate_rmse_m":
            float(
                selected_history[
                    "val_recursive_ate_rmse_m"
                ]
            ),

        "local_translation_rmse_m":
            float(
                selected_history[
                    "val_translation_rmse_m"
                ]
            ),
    },

    "final_test": {
        "pairs":
            int(
                final_summary[
                    "final_test_pairs"
                ]
            ),

        "recursive_ate_rmse_m":
            float(
                final_summary[
                    "recursive_ate_rmse_m"
                ]
            ),

        "local_translation_rmse_m":
            float(
                final_summary[
                    "local_translation_rmse_m"
                ]
            ),

        "total_gt_distance_m":
            float(
                final_summary[
                    "total_gt_distance_m"
                ]
            ),

        "total_pred_distance_m":
            float(
                final_summary[
                    "total_pred_distance_m"
                ]
            ),

        "pred_gt_distance_ratio":
            float(
                final_summary[
                    "overall_pred_gt_distance_ratio"
                ]
            ),
    },

    "drift_investigation":
        "closed",

    "canonical_model_changed_after_final_test":
        False,

    "status":
        "FINAL RESULTS PACKAGE — CANONICAL FROZEN MODEL",
}


OUT_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# Report
# ============================================================

print()
print("=" * 100)
print("THESIS FINAL PERFORMANCE PACKAGE")
print("=" * 100)

print(
    "Canonical checkpoint SHA256 :",
    freeze["sha256"]
)

print(
    "Selected epoch              :",
    selected_epoch
)

print(
    "Validation recursive ATE    :",
    f"{summary['validation']['recursive_ate_rmse_m']:.6f} m"
)

print(
    "Final-test recursive ATE    :",
    f"{summary['final_test']['recursive_ate_rmse_m']:.6f} m"
)

print(
    "Final-test local RMSE       :",
    f"{summary['final_test']['local_translation_rmse_m']:.6f} m"
)

print(
    "Final-test GT distance      :",
    f"{summary['final_test']['total_gt_distance_m']:.2f} m"
)

print(
    "Final-test Pred/GT ratio    :",
    f"{summary['final_test']['pred_gt_distance_ratio']:.6f}"
)

print()
print("Saved:")
print(" ", OUT_CSV)
print(" ", OUT_JSON)

print()
print(
    "PASS: canonical thesis performance "
    "table created."
)

print("=" * 100)
