# compare_motion_cues.py

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def read_csv(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def main():
    os.makedirs("results", exist_ok=True)

    files = {
        "Base PINN+IMU+EKF": "results/gnss_denied_results.csv",
        "LiDAR cue": "results/lidar_odometry_ekf_results.csv",
        "Visual cue": "results/visual_odometry_ekf_results.csv",
    }

    all_rows = []

    for method, path in files.items():
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue

        rows = read_csv(path)

        for row in rows:
            experiment = row["experiment"]

            if "No outage" in experiment or "normal" in experiment:
                scenario = "Normal GNSS"
            elif "50-frame" in experiment:
                scenario = "50-frame outage"
            elif "100-frame" in experiment:
                scenario = "100-frame outage"
            elif "Full" in experiment or "full" in experiment:
                scenario = "Full GNSS-denied"
            elif "20-frame" in experiment:
                scenario = "20-frame outage"
            else:
                scenario = experiment

            all_rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "ATE_m": to_float(row, "ATE_m"),
                    "RPE_m": to_float(row, "RPE_m"),
                    "Mean_m": to_float(row, "Mean_m"),
                    "Median_m": to_float(row, "Median_m"),
                    "Max_m": to_float(row, "Max_m"),
                    "P95_m": to_float(row, "P95_m"),
                }
            )

    out_csv = "results/final_motion_cue_comparison.csv"

    fieldnames = [
        "method",
        "scenario",
        "ATE_m",
        "RPE_m",
        "Mean_m",
        "Median_m",
        "Max_m",
        "P95_m",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nFinal comparison")
    print("=" * 95)
    print(f"{'Method':24s} {'Scenario':22s} {'ATE':>8s} {'RPE':>8s} {'Mean':>8s} {'Max':>8s} {'P95':>8s}")
    print("-" * 95)

    for row in all_rows:
        print(
            f"{row['method']:24s} "
            f"{row['scenario']:22s} "
            f"{row['ATE_m']:8.3f} "
            f"{row['RPE_m']:8.3f} "
            f"{row['Mean_m']:8.3f} "
            f"{row['Max_m']:8.3f} "
            f"{row['P95_m']:8.3f}"
        )

    # Plot ATE comparison
    scenarios = [
        "Normal GNSS",
        "50-frame outage",
        "100-frame outage",
        "Full GNSS-denied",
    ]

    methods = [
        "Base PINN+IMU+EKF",
        "LiDAR cue",
        "Visual cue",
    ]

    x = np.arange(len(scenarios))
    width = 0.25

    plt.figure(figsize=(11, 5))

    for i, method in enumerate(methods):
        vals = []

        for scenario in scenarios:
            match = [
                r for r in all_rows
                if r["method"] == method and r["scenario"] == scenario
            ]

            if len(match) == 0:
                vals.append(np.nan)
            else:
                vals.append(match[0]["ATE_m"])

        plt.bar(x + (i - 1) * width, vals, width, label=method)

    plt.ylabel("ATE (m)")
    plt.title("Final Localization Comparison under GNSS Outages")
    plt.xticks(x, scenarios, rotation=15, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = "results/final_motion_cue_ate_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.show()

    print("\nSaved:")
    print(out_csv)
    print(plot_path)


if __name__ == "__main__":
    main()