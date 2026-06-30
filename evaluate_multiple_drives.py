# evaluate_multiple_drives.py

import os
import csv
import numpy as np
import torch

from config import Config
from src.models import FusionModel
from src.raw_kitti_utils import load_raw_sequence

from evaluate_baselines import (
    summarize,
    integrate_pinn_learned_yaw,
    integrate_pinn_imu_yaw,
    integrate_oracle_yaw,
    ekf_only_baseline,
    pinn_imu_ekf,
)


def drive_name_from_path(path):
    parts = path.replace("\\", "/").split("/")
    for p in parts:
        if "drive_" in p and "_sync" in p:
            return p
    return parts[-1]


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model_path = "models/pinn_delta_physics.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Run python train_physics_loss.py first."
        )

    model = FusionModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    sequence_dirs = getattr(config, "raw_sequence_dirs", [config.seq_dir])

    all_rows = []

    for seq_dir in sequence_dirs:
        print("\n" + "=" * 100)
        print(f"Evaluating drive: {seq_dir}")
        print("=" * 100)

        drive_name = drive_name_from_path(seq_dir)

        original_seq_dir = config.seq_dir
        config.seq_dir = seq_dir

        try:
            states, lidar_pts, cam_imgs, imu_data = load_raw_sequence(config.seq_dir, config)
        except Exception as e:
            print(f"Skipping {seq_dir} because of error: {e}")
            config.seq_dir = original_seq_dir
            continue

        config.seq_dir = original_seq_dir

        drive_results = []

        # ------------------------------------------------------------
        # 1. Pure PINN learned yaw
        # ------------------------------------------------------------
        traj_learned = integrate_pinn_learned_yaw(
            model,
            states,
            lidar_pts,
            cam_imgs,
            imu_data,
            config,
            device,
        )
        gt_xy = states[: len(traj_learned), :2]
        pred_xy = traj_learned[:, :2]
        row = summarize("Pure PINN learned yaw", pred_xy, gt_xy)
        row["drive"] = drive_name
        row["frames"] = len(states)
        drive_results.append(row)

        # ------------------------------------------------------------
        # 2. PINN + IMU yaw
        # ------------------------------------------------------------
        traj_imu = integrate_pinn_imu_yaw(
            model,
            states,
            lidar_pts,
            cam_imgs,
            imu_data,
            config,
            device,
        )
        gt_xy = states[: len(traj_imu), :2]
        pred_xy = traj_imu[:, :2]
        row = summarize("PINN + IMU yaw", pred_xy, gt_xy)
        row["drive"] = drive_name
        row["frames"] = len(states)
        drive_results.append(row)

        # ------------------------------------------------------------
        # 3. PINN + IMU yaw + EKF/OXTS
        # ------------------------------------------------------------
        pred_ekf = pinn_imu_ekf(
            model,
            states,
            lidar_pts,
            cam_imgs,
            imu_data,
            config,
            device,
            correction_interval=10,
        )
        gt_xy = states[: len(pred_ekf), :2]
        row = summarize("PINN + IMU yaw + EKF/OXTS", pred_ekf, gt_xy)
        row["drive"] = drive_name
        row["frames"] = len(states)
        drive_results.append(row)

        # ------------------------------------------------------------
        # 4. EKF-only baseline
        # ------------------------------------------------------------
        pred_ekf_only = ekf_only_baseline(
            states,
            imu_data,
            config,
            correction_interval=10,
        )
        gt_xy = states[: len(pred_ekf_only), :2]
        row = summarize("EKF-only IMU + OXTS", pred_ekf_only, gt_xy)
        row["drive"] = drive_name
        row["frames"] = len(states)
        drive_results.append(row)

        # ------------------------------------------------------------
        # 5. Oracle yaw diagnostic
        # ------------------------------------------------------------
        traj_oracle = integrate_oracle_yaw(
            model,
            states,
            lidar_pts,
            cam_imgs,
            imu_data,
            config,
            device,
        )
        gt_xy = states[: len(traj_oracle), :2]
        pred_xy = traj_oracle[:, :2]
        row = summarize("Oracle yaw diagnostic", pred_xy, gt_xy)
        row["drive"] = drive_name
        row["frames"] = len(states)
        drive_results.append(row)

        print("\nDrive result")
        print("-" * 110)
        print(
            f"{'Method':35s} {'ATE':>10s} {'RPE':>10s} "
            f"{'Mean':>10s} {'Median':>10s} {'Max':>10s} {'P95':>10s}"
        )
        print("-" * 110)

        for r in drive_results:
            print(
                f"{r['method']:35s} "
                f"{r['ATE_m']:10.3f} "
                f"{r['RPE_m']:10.3f} "
                f"{r['Mean_m']:10.3f} "
                f"{r['Median_m']:10.3f} "
                f"{r['Max_m']:10.3f} "
                f"{r['P95_m']:10.3f}"
            )

        all_rows.extend(drive_results)

    if len(all_rows) == 0:
        raise RuntimeError("No drives were evaluated successfully.")

    os.makedirs("results", exist_ok=True)

    full_csv = "results/multi_drive_baseline_results.csv"

    fieldnames = [
        "drive",
        "frames",
        "method",
        "ATE_m",
        "RPE_m",
        "Mean_m",
        "Median_m",
        "Max_m",
        "P95_m",
    ]

    with open(full_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # ------------------------------------------------------------
    # Summary: mean/std across drives for each method
    # ------------------------------------------------------------
    methods = sorted(set(row["method"] for row in all_rows))
    summary_rows = []

    for method in methods:
        rows = [r for r in all_rows if r["method"] == method]

        summary = {
            "method": method,
            "num_drives": len(rows),
            "ATE_mean": float(np.mean([r["ATE_m"] for r in rows])),
            "ATE_std": float(np.std([r["ATE_m"] for r in rows])),
            "RPE_mean": float(np.mean([r["RPE_m"] for r in rows])),
            "RPE_std": float(np.std([r["RPE_m"] for r in rows])),
            "Max_mean": float(np.mean([r["Max_m"] for r in rows])),
            "Max_std": float(np.std([r["Max_m"] for r in rows])),
            "P95_mean": float(np.mean([r["P95_m"] for r in rows])),
            "P95_std": float(np.std([r["P95_m"] for r in rows])),
        }

        summary_rows.append(summary)

    summary_csv = "results/multi_drive_summary.csv"

    summary_fields = [
        "method",
        "num_drives",
        "ATE_mean",
        "ATE_std",
        "RPE_mean",
        "RPE_std",
        "Max_mean",
        "Max_std",
        "P95_mean",
        "P95_std",
    ]

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 110)
    print("Multi-drive summary")
    print("=" * 110)
    print(
        f"{'Method':35s} {'Drives':>8s} {'ATE mean':>10s} {'ATE std':>10s} "
        f"{'RPE mean':>10s} {'RPE std':>10s}"
    )
    print("-" * 110)

    for r in summary_rows:
        print(
            f"{r['method']:35s} "
            f"{r['num_drives']:8d} "
            f"{r['ATE_mean']:10.3f} "
            f"{r['ATE_std']:10.3f} "
            f"{r['RPE_mean']:10.3f} "
            f"{r['RPE_std']:10.3f}"
        )

    print(f"\nSaved full results to: {full_csv}")
    print(f"Saved summary to: {summary_csv}")


if __name__ == "__main__":
    main()