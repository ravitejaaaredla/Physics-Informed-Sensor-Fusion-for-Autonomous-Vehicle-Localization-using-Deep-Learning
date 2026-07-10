import sys
from pathlib import Path
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from src.data.kitti_dataset import KITTIRawDataset
from src.models.fusion_pinn import FusionPINN
from src.localization.ekf import FourStateEKF
from src.localization.coordinates import xy_to_latlon
from src.evaluation.metrics import compute_ate, compute_rpe


def should_use_gnss(frame_index, gnss_denied_start, gnss_denied_len):
    if gnss_denied_start < 0 or gnss_denied_len <= 0:
        return True

    denied_end = gnss_denied_start + gnss_denied_len

    if gnss_denied_start <= frame_index < denied_end:
        return False

    return True


def save_metrics(output_dir, ate, rpe, gnss_denied_start, gnss_denied_len):
    metrics_file = output_dir / "metrics.csv"

    rows = [
        ["metric", "value"],
        ["mean_ate_m", float(np.mean(ate))],
        ["median_ate_m", float(np.median(ate))],
        ["max_ate_m", float(np.max(ate))],
        ["ate_95_percentile_m", float(np.percentile(ate, 95))],
        ["mean_rpe_m", float(np.mean(rpe)) if len(rpe) > 0 else 0.0],
        ["median_rpe_m", float(np.median(rpe)) if len(rpe) > 0 else 0.0],
        ["max_rpe_m", float(np.max(rpe)) if len(rpe) > 0 else 0.0],
        ["rpe_95_percentile_m", float(np.percentile(rpe, 95)) if len(rpe) > 0 else 0.0],
        ["gnss_denied_start", gnss_denied_start],
        ["gnss_denied_len", gnss_denied_len],
    ]

    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("Saved metrics:", metrics_file)


def save_latlon_csv(output_dir, trajectory_rows):
    csv_file = output_dir / "trajectory_latlon.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "frame",
                "predicted_x",
                "predicted_y",
                "predicted_latitude",
                "predicted_longitude",
                "ground_truth_x",
                "ground_truth_y",
                "ground_truth_latitude",
                "ground_truth_longitude",
                "use_gnss",
            ]
        )

        writer.writerows(trajectory_rows)

    print("Saved trajectory lat/lon CSV:", csv_file)


def plot_trajectory(output_dir, predicted_xy, ground_truth_xy, title):
    output_file = output_dir / "trajectory.png"

    plt.figure(figsize=(8, 6))

    plt.plot(
        ground_truth_xy[:, 0],
        ground_truth_xy[:, 1],
        label="Ground truth OXTS",
        linewidth=2,
    )

    plt.plot(
        predicted_xy[:, 0],
        predicted_xy[:, 1],
        label="PINN + EKF prediction",
        linewidth=2,
    )

    plt.scatter(
        ground_truth_xy[0, 0],
        ground_truth_xy[0, 1],
        s=80,
        label="Start",
    )

    plt.scatter(
        ground_truth_xy[-1, 0],
        ground_truth_xy[-1, 1],
        s=80,
        label="End",
    )

    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

    print("Saved trajectory plot:", output_file)


def plot_ate(output_dir, ate, gnss_denied_start, gnss_denied_len):
    output_file = output_dir / "ate.png"

    plt.figure(figsize=(10, 4))
    plt.plot(ate, label="ATE")

    if gnss_denied_start >= 0 and gnss_denied_len > 0:
        start = gnss_denied_start
        end = gnss_denied_start + gnss_denied_len
        plt.axvspan(start, end, alpha=0.2, label="GNSS denied interval")

    plt.xlabel("Frame")
    plt.ylabel("ATE [m]")
    plt.title("Absolute Trajectory Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

    print("Saved ATE plot:", output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnss_denied_start", type=int, default=-1)
    parser.add_argument("--gnss_denied_len", type=int, default=0)
    args = parser.parse_args()

    print("=" * 80)
    print("09 - EVALUATE LOCALIZATION")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)
    print("Test drive:", Config.test_sequence_dir)

    dataset = KITTIRawDataset(
        [Config.test_sequence_dir],
        max_frames=Config.max_frames,
    )

    print("Number of test samples:", len(dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_path = Config.checkpoint_dir / "model_best.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train first using scripts/08_train_model.py"
        )

    model = FusionPINN(Config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Loaded model:", model_path)

    first_sample = dataset[0]
    first_aux = first_sample["aux"]

    origin_lat = first_aux["lat_now"]
    origin_lon = first_aux["lon_now"]

    ekf = FourStateEKF(Config)

    ekf.initialize(
        x=first_aux["x_now"],
        y=first_aux["y_now"],
        velocity=first_aux["speed_now"],
        yaw=first_aux["yaw_now"],
    )

    print("\nInitial EKF state:")
    print(ekf.x)

    print("\nOrigin latitude/longitude:")
    print(origin_lat, origin_lon)

    predicted_positions = []
    ground_truth_positions = []
    trajectory_rows = []

    for frame_index in range(len(dataset)):
        sample = dataset[frame_index]

        camera = sample["camera"].unsqueeze(0).to(device)
        lidar_bev = sample["lidar_bev"].unsqueeze(0).to(device)
        imu = sample["imu"].unsqueeze(0).to(device)
        aux = sample["aux"]

        with torch.no_grad():
            predicted_motion = model(camera, lidar_bev, imu)

        predicted_motion = predicted_motion[0].detach().cpu().numpy()

        dx_body = float(predicted_motion[0])
        dy_body = float(predicted_motion[1])
        dv = float(predicted_motion[2])
        dyaw = float(predicted_motion[3])

        state = ekf.predict(dx_body, dy_body, dv, dyaw)

        use_gnss = should_use_gnss(
            frame_index,
            args.gnss_denied_start,
            args.gnss_denied_len,
        )

        if use_gnss:
            state = ekf.update_gnss(
                aux["x_next"],
                aux["y_next"],
            )

        pred_x = float(state[0])
        pred_y = float(state[1])

        gt_x = float(aux["x_next"])
        gt_y = float(aux["y_next"])

        pred_lat, pred_lon = xy_to_latlon(
            pred_x,
            pred_y,
            origin_lat,
            origin_lon,
        )

        gt_lat = float(aux["lat_next"])
        gt_lon = float(aux["lon_next"])

        predicted_positions.append([pred_x, pred_y])
        ground_truth_positions.append([gt_x, gt_y])

        trajectory_rows.append(
            [
                frame_index,
                pred_x,
                pred_y,
                pred_lat,
                pred_lon,
                gt_x,
                gt_y,
                gt_lat,
                gt_lon,
                use_gnss,
            ]
        )

    predicted_xy = np.asarray(predicted_positions, dtype=np.float64)
    ground_truth_xy = np.asarray(ground_truth_positions, dtype=np.float64)

    ate = compute_ate(predicted_xy, ground_truth_xy)
    rpe = compute_rpe(predicted_xy, ground_truth_xy)

    print("\nEvaluation results")
    print("-" * 80)
    print("Mean ATE [m]:", float(np.mean(ate)))
    print("Median ATE [m]:", float(np.median(ate)))
    print("Max ATE [m]:", float(np.max(ate)))
    print("95th Percentile ATE [m]:", float(np.percentile(ate, 95)))

    if len(rpe) > 0:
        print("Mean RPE [m]:", float(np.mean(rpe)))
        print("Median RPE [m]:", float(np.median(rpe)))
        print("Max RPE [m]:", float(np.max(rpe)))
        print("95th Percentile RPE [m]:", float(np.percentile(rpe, 95)))

    if args.gnss_denied_start >= 0 and args.gnss_denied_len > 0:
        output_dir = Config.eval_dir / "gnss_denied"
        title = "PINN-EKF localization with GNSS-denied interval"
    else:
        output_dir = Config.eval_dir / "normal"
        title = "PINN-EKF localization with OXTS/GNSS correction"

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_trajectory(output_dir, predicted_xy, ground_truth_xy, title)
    plot_ate(output_dir, ate, args.gnss_denied_start, args.gnss_denied_len)
    save_metrics(output_dir, ate, rpe, args.gnss_denied_start, args.gnss_denied_len)
    save_latlon_csv(output_dir, trajectory_rows)

    print("\nDONE")
    print("Output folder:")
    print(output_dir)


if __name__ == "__main__":
    main()

