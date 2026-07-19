import subprocess
import csv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "runs" / "eval" / "gnss_denied"

SCALE_VALUES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

results = {}
for scale in SCALE_VALUES:
    cmd = (
        f"python scripts/evaluate_localization.py "
        f"--start_frame 200 --num_frames 50 "
        f"--gnss_denied_start 200 --gnss_denied_len 50 "
        f"--use_imu_dyaw --use_gt_motion "
        f"--imu_scale {scale}"
    )
    print(f"\n>>> Running scale {scale} ...")
    subprocess.run(cmd, shell=True, check=True)

    # Read metrics
    metrics_file = OUTPUT_DIR / "metrics.csv"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == "mean_ate_m":
                    results[scale] = float(row[1])
                    print(f"Scale {scale}: Mean ATE = {results[scale]:.3f} m")
                    break

# Find best
best_scale = min(results, key=results.get)
print("\n" + "=" * 60)
print("BEST RESULT")
print(f"IMU_YAW_SCALE = {best_scale}  →  Mean ATE = {results[best_scale]:.3f} m")
print("=" * 60)