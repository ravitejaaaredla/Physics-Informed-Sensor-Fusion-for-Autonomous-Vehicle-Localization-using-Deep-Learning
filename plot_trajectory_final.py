# plot_trajectory_final.py

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from src.models import FusionModel
from src.raw_kitti_utils import load_raw_sequence


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def ate(pred_xy, gt_xy):
    n = min(len(pred_xy), len(gt_xy))
    pred_xy = pred_xy[:n]
    gt_xy = gt_xy[:n]
    return float(np.sqrt(np.mean(np.sum((pred_xy - gt_xy) ** 2, axis=1))))


def rpe(pred_xy, gt_xy):
    n = min(len(pred_xy), len(gt_xy))
    pred_xy = pred_xy[:n]
    gt_xy = gt_xy[:n]

    if n < 2:
        return float("nan")

    pred_rel = pred_xy[1:] - pred_xy[:-1]
    gt_rel = gt_xy[1:] - gt_xy[:-1]
    return float(np.sqrt(np.mean(np.sum((pred_rel - gt_rel) ** 2, axis=1))))


def integrate_delta_model(model, states, lidar_pts, cam_imgs, imu_data, config, device):
    model.eval()

    pred_states = []

    x, y, v, yaw = states[0]
    pred_states.append([x, y, v, yaw])

    for end in range(config.window_size - 1, len(states) - 1):
        start = end - config.window_size + 1

        imu_win = imu_data[start:end + 1]
        lidar = lidar_pts[end]
        cam = cam_imgs[end]

        imu_t = torch.tensor(imu_win, dtype=torch.float32).unsqueeze(0).to(device)
        lidar_t = torch.tensor(lidar, dtype=torch.float32).unsqueeze(0).to(device)
        cam_t = (
            torch.tensor(cam, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            delta = model(imu_t, lidar_t, cam_t).cpu().numpy()[0]

        # --------------------------------------------------
        # DEBUG ONLY: Ground-truth scale calibration
        # This checks whether the model's main problem is
        # wrong step size or wrong movement direction.
        #
        # Do NOT report this as final fair thesis result.
        # --------------------------------------------------
        gt_step = states[end + 1, :2] - states[end, :2]
        gt_dist = np.linalg.norm(gt_step)

        pred_dist = np.linalg.norm(delta[:2]) + 1e-8

        scale = gt_dist / pred_dist
        scale = np.clip(scale, 0.2, 5.0)

        delta[0] *= scale
        delta[1] *= scale

        dx_b, dy_b, dv, dyaw = delta

        c = np.cos(yaw)
        s = np.sin(yaw)

        dx_w = c * dx_b - s * dy_b
        dy_w = s * dx_b + c * dy_b

        x = x + dx_w
        y = y + dy_w
        v = max(0.0, v + dv)
        # DEBUG ONLY:
        # Use ground-truth yaw to test whether predicted dyaw causes drift.
        gyro_z = imu_win[-1, 5]
        yaw = wrap_angle(yaw + gyro_z * config.dt)

        pred_states.append([x, y, v, yaw])

    return np.asarray(pred_states, dtype=np.float32)


def plot_all(gt_xy, pred_xy, save_prefix):
    os.makedirs("results", exist_ok=True)

    n = min(len(gt_xy), len(pred_xy))
    gt_xy = gt_xy[:n]
    pred_xy = pred_xy[:n]

    error = np.linalg.norm(pred_xy - gt_xy, axis=1)

    print(f"ATE: {ate(pred_xy, gt_xy):.3f} m")
    print(f"RPE: {rpe(pred_xy, gt_xy):.3f} m")
    print(f"Mean error: {np.mean(error):.3f} m")
    print(f"Median error: {np.median(error):.3f} m")
    print(f"Max error: {np.max(error):.3f} m")
    print(f"95th percentile error: {np.percentile(error, 95):.3f} m")

    plt.figure(figsize=(8, 6))
    plt.plot(gt_xy[:, 0], gt_xy[:, 1], "k-", linewidth=2.5, label="Ground Truth")
    plt.plot(
        pred_xy[:, 0],
        pred_xy[:, 1],
        linewidth=2.0,
        label="PINN Delta Physics + GT Scale Debug",
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Trajectory Comparison - Scale Debug")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{save_prefix}_trajectory_scale_debug.png", dpi=200)
    plt.show()

    start = min(50, n - 2)
    end = min(150, n)

    plt.figure(figsize=(8, 6))
    plt.plot(
        gt_xy[start:end, 0],
        gt_xy[start:end, 1],
        "k-",
        linewidth=2.5,
        label="Ground Truth",
    )
    plt.plot(
        pred_xy[start:end, 0],
        pred_xy[start:end, 1],
        linewidth=2.0,
        label="PINN Delta Physics + GT Scale Debug",
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"Zoomed Trajectory Frames {start}-{end} - Scale Debug")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{save_prefix}_trajectory_zoom_scale_debug.png", dpi=200)
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(n), error, linewidth=1.5)
    plt.xlabel("Frame")
    plt.ylabel("Position Error (m)")
    plt.title("Position Error over Time - Scale Debug")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{save_prefix}_error_over_time_scale_debug.png", dpi=200)
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.hist(error, bins=30, alpha=0.8)
    plt.xlabel("Position Error (m)")
    plt.ylabel("Count")
    plt.title("Error Distribution - Scale Debug")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{save_prefix}_error_histogram_scale_debug.png", dpi=200)
    plt.show()

    sorted_error = np.sort(error)
    cdf = np.arange(1, len(sorted_error) + 1) / len(sorted_error)

    plt.figure(figsize=(7, 4))
    plt.plot(sorted_error, cdf, linewidth=2)
    plt.xlabel("Position Error (m)")
    plt.ylabel("CDF")
    plt.title("Cumulative Error Distribution - Scale Debug")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{save_prefix}_error_cdf_scale_debug.png", dpi=200)
    plt.show()


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    states, lidar_pts, cam_imgs, imu_data = load_raw_sequence(config.seq_dir, config)

    model_path = "models/pinn_delta_physics.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Run this first: python train_physics_loss.py"
        )

    model = FusionModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    pred_states = integrate_delta_model(
        model=model,
        states=states,
        lidar_pts=lidar_pts,
        cam_imgs=cam_imgs,
        imu_data=imu_data,
        config=config,
        device=device,
    )

    gt_xy = states[: len(pred_states), :2]
    pred_xy = pred_states[:, :2]

    np.save("results/pinn_delta_pred_xy_scale_debug.npy", pred_xy)
    np.save("results/gt_xy_scale_debug.npy", gt_xy)

    plot_all(gt_xy, pred_xy, "pinn_delta_physics")


if __name__ == "__main__":
    main()