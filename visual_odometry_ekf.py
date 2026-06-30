# visual_odometry_ekf.py

import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    raise ImportError("OpenCV is not installed. Install it with: pip install opencv-python")

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


def compute_stats(pred_xy, gt_xy):
    n = min(len(pred_xy), len(gt_xy))
    pred_xy = pred_xy[:n]
    gt_xy = gt_xy[:n]

    error = np.linalg.norm(pred_xy - gt_xy, axis=1)

    return {
        "ATE_m": ate(pred_xy, gt_xy),
        "RPE_m": rpe(pred_xy, gt_xy),
        "Mean_m": float(np.mean(error)),
        "Median_m": float(np.median(error)),
        "Max_m": float(np.max(error)),
        "P95_m": float(np.percentile(error, 95)),
    }


def predict_delta(model, imu_win, lidar, cam, device):
    model.eval()

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

    return delta.astype(np.float32)


def to_gray_uint8(img):
    """
    Input img from loader: RGB float32 [0,1]
    Output: grayscale uint8
    """
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    return gray


def estimate_visual_motion(prev_img, curr_img, pinn_dx, pinn_dy):
    """
    Lightweight visual odometry cue using ORB + Essential Matrix.

    This is NOT full metric visual odometry because monocular scale is unknown.
    We use visual pose direction and scale it using the PINN displacement magnitude.

    Returns:
        dx_body, dy_body, quality
    """
    prev_gray = to_gray_uint8(prev_img)
    curr_gray = to_gray_uint8(curr_img)

    h, w = prev_gray.shape

    orb = cv2.ORB_create(
        nfeatures=1200,
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=20,
    )

    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None:
        return 0.0, 0.0, 0.0

    if len(kp1) < 30 or len(kp2) < 30:
        return 0.0, 0.0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    if len(matches) < 30:
        return 0.0, 0.0, 0.0

    matches = sorted(matches, key=lambda m: m.distance)

    # Keep best matches
    keep_n = min(200, len(matches))
    matches = matches[:keep_n]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Approximate intrinsics after resizing to config.camera_img_w x config.camera_img_h.
    # This is acceptable for a visual motion cue, not for final calibrated VO.
    fx = 0.55 * w
    fy = 1.40 * h
    cx = w / 2.0
    cy = h / 2.0

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )

    if E is None:
        return 0.0, 0.0, 0.0

    try:
        _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K)
    except cv2.error:
        return 0.0, 0.0, 0.0

    if pose_mask is None:
        return 0.0, 0.0, 0.0

    inliers = int(np.sum(pose_mask > 0))

    if inliers < 25:
        return 0.0, 0.0, 0.0

    t = t.reshape(3)

    norm_t = np.linalg.norm(t) + 1e-8
    t = t / norm_t

    # Camera coordinates:
    # z = forward, x = right.
    # Vehicle/body coordinates used here:
    # x_body = forward, y_body = left.
    direction_dx = float(t[2])
    direction_dy = float(-t[0])

    direction_norm = np.sqrt(direction_dx ** 2 + direction_dy ** 2) + 1e-8
    direction_dx /= direction_norm
    direction_dy /= direction_norm

    pinn_step = np.sqrt(pinn_dx ** 2 + pinn_dy ** 2)

    # If PINN step is unreasonable, reject visual cue.
    if pinn_step <= 1e-6 or pinn_step > 5.0:
        return 0.0, 0.0, 0.0

    dx_body = direction_dx * pinn_step
    dy_body = direction_dy * pinn_step

    quality = min(1.0, inliers / 120.0)

    return dx_body, dy_body, quality


class HybridVisualEKF:
    """
    State:
        x = [px, py, v, yaw]

    Prediction:
        PINN gives translation.
        Visual motion cue gives monocular direction.
        IMU gyro_z propagates yaw.

    Correction:
        sparse OXTS/GNSS position update.
    """

    def __init__(self, model, config, device, visual_weight=0.10):
        self.model = model
        self.config = config
        self.device = device
        self.dt = config.dt
        self.visual_weight = visual_weight

        self.x = np.zeros(4, dtype=np.float32)

        self.P = np.diag([1.0, 1.0, 0.5, 0.1]).astype(np.float32)

        q_pos = getattr(config, "process_noise_pos", 0.5)
        q_vel = getattr(config, "process_noise_vel", 0.5)
        q_yaw = getattr(config, "process_noise_yaw", 0.1)

        self.Q = np.diag(
            [
                q_pos ** 2,
                q_pos ** 2,
                q_vel ** 2,
                q_yaw ** 2,
            ]
        ).astype(np.float32)

        gnss_noise = getattr(config, "gnss_noise_std", 1.0)

        self.R_gnss = np.diag(
            [
                gnss_noise ** 2,
                gnss_noise ** 2,
            ]
        ).astype(np.float32)

    def initialize(self, initial_state):
        self.x[:] = initial_state.astype(np.float32)

    def predict(self, imu_win, lidar, prev_cam, curr_cam):
        delta = predict_delta(
            model=self.model,
            imu_win=imu_win,
            lidar=lidar,
            cam=curr_cam,
            device=self.device,
        )

        pinn_dx = float(delta[0])
        pinn_dy = float(delta[1])
        pinn_dv = float(delta[2])

        visual_dx, visual_dy, visual_quality = estimate_visual_motion(
            prev_img=prev_cam,
            curr_img=curr_cam,
            pinn_dx=pinn_dx,
            pinn_dy=pinn_dy,
        )

        use_visual = visual_quality > 0.20

        if use_visual:
            w = self.visual_weight * visual_quality
            dx_body = (1.0 - w) * pinn_dx + w * visual_dx
            dy_body = (1.0 - w) * pinn_dy + w * visual_dy
        else:
            dx_body = pinn_dx
            dy_body = pinn_dy

        gyro_z = float(imu_win[-1, 5])
        dyaw_imu = gyro_z * self.dt

        px, py, v, yaw = self.x

        c = np.cos(yaw)
        s = np.sin(yaw)

        dx_world = c * dx_body - s * dy_body
        dy_world = s * dx_body + c * dy_body

        px_new = px + dx_world
        py_new = py + dy_world
        v_new = max(0.0, v + pinn_dv)
        yaw_new = wrap_angle(yaw + dyaw_imu)

        self.x = np.array(
            [
                px_new,
                py_new,
                v_new,
                yaw_new,
            ],
            dtype=np.float32,
        )

        F = np.eye(4, dtype=np.float32)
        F[0, 3] = -s * dx_body - c * dy_body
        F[1, 3] = c * dx_body - s * dy_body

        self.P = F @ self.P @ F.T + self.Q

        return {
            "use_visual": use_visual,
            "visual_quality": visual_quality,
            "pinn_dx": pinn_dx,
            "pinn_dy": pinn_dy,
            "visual_dx": visual_dx,
            "visual_dy": visual_dy,
            "dx_body": dx_body,
            "dy_body": dy_body,
        }

    def update_gnss(self, z_xy):
        z = np.asarray(z_xy, dtype=np.float32)

        H = np.zeros((2, 4), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        residual = z - H @ self.x
        S = H @ self.P @ H.T + self.R_gnss
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ residual
        self.x[3] = wrap_angle(self.x[3])

        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ H) @ self.P


def is_inside_outage(frame_idx, outage_start, outage_length):
    if outage_length is None:
        return True

    outage_end = outage_start + outage_length
    return outage_start <= frame_idx < outage_end


def run_visual_ekf(
    model,
    states,
    lidar_pts,
    cam_imgs,
    imu_data,
    config,
    device,
    correction_interval=10,
    outage_start_frame=10_000_000,
    outage_length=0,
    visual_weight=0.10,
):
    ekf = HybridVisualEKF(
        model=model,
        config=config,
        device=device,
        visual_weight=visual_weight,
    )

    ekf.initialize(states[0])

    estimates = []
    estimates.append(ekf.x[:2].copy())

    corrections_used = 0
    corrections_blocked = 0
    visual_used = 0
    visual_failed = 0

    for end in range(config.window_size - 1, len(states) - 1):
        start = end - config.window_size + 1

        imu_win = imu_data[start:end + 1]
        lidar = lidar_pts[end]
        prev_cam = cam_imgs[end - 1]
        curr_cam = cam_imgs[end]

        info = ekf.predict(
            imu_win=imu_win,
            lidar=lidar,
            prev_cam=prev_cam,
            curr_cam=curr_cam,
        )

        if info["use_visual"]:
            visual_used += 1
        else:
            visual_failed += 1

        should_correct = end % correction_interval == 0

        if should_correct:
            denied = is_inside_outage(
                frame_idx=end,
                outage_start=outage_start_frame,
                outage_length=outage_length,
            )

            if denied:
                corrections_blocked += 1
            else:
                ekf.update_gnss(states[end + 1, :2])
                corrections_used += 1

        estimates.append(ekf.x[:2].copy())

    pred_xy = np.asarray(estimates, dtype=np.float32)
    gt_xy = states[: len(pred_xy), :2]

    return pred_xy, gt_xy, {
        "corrections_used": corrections_used,
        "corrections_blocked": corrections_blocked,
        "visual_used": visual_used,
        "visual_failed": visual_failed,
    }


def plot_trajectories(gt_xy, trajectories, save_path):
    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(9, 7))

    plt.plot(
        gt_xy[:, 0],
        gt_xy[:, 1],
        "k-",
        linewidth=3,
        label="Ground Truth / OXTS",
    )

    for name, pred_xy in trajectories.items():
        n = min(len(gt_xy), len(pred_xy))
        plt.plot(pred_xy[:n, 0], pred_xy[:n, 1], linewidth=1.8, label=name)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Visual Motion Cue + PINN + IMU + EKF")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


def plot_ate_bar(results, save_path):
    os.makedirs("results", exist_ok=True)

    labels = [r["experiment"] for r in results]
    values = [r["ATE_m"] for r in results]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.ylabel("ATE (m)")
    plt.title("Visual Motion Cue Integration Results")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model_path = "models/pinn_delta_physics.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Run first: python train_physics_loss.py"
        )

    model = FusionModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    states, lidar_pts, cam_imgs, imu_data = load_raw_sequence(config.seq_dir, config)

    experiments = [
        {
            "name": "Visual+GNSS normal",
            "outage_start": 10_000_000,
            "outage_length": 0,
        },
        {
            "name": "Visual 50-frame GNSS outage",
            "outage_start": 80,
            "outage_length": 50,
        },
        {
            "name": "Visual 100-frame GNSS outage",
            "outage_start": 80,
            "outage_length": 100,
        },
        {
            "name": "Visual full GNSS-denied",
            "outage_start": 0,
            "outage_length": None,
        },
    ]

    results = []
    trajectories = {}

    for exp in experiments:
        print("\n" + "=" * 80)
        print(exp["name"])
        print("=" * 80)

        pred_xy, gt_xy, info = run_visual_ekf(
            model=model,
            states=states,
            lidar_pts=lidar_pts,
            cam_imgs=cam_imgs,
            imu_data=imu_data,
            config=config,
            device=device,
            correction_interval=10,
            outage_start_frame=exp["outage_start"],
            outage_length=exp["outage_length"],
            visual_weight=0.10,
        )

        stats = compute_stats(pred_xy, gt_xy)

        row = {
            "experiment": exp["name"],
            **stats,
            **info,
        }

        results.append(row)
        trajectories[exp["name"]] = pred_xy

        print(f"ATE: {stats['ATE_m']:.3f} m")
        print(f"RPE: {stats['RPE_m']:.3f} m")
        print(f"Mean error: {stats['Mean_m']:.3f} m")
        print(f"Median error: {stats['Median_m']:.3f} m")
        print(f"Max error: {stats['Max_m']:.3f} m")
        print(f"95th percentile error: {stats['P95_m']:.3f} m")
        print(f"GNSS/OXTS corrections used: {info['corrections_used']}")
        print(f"GNSS/OXTS corrections blocked: {info['corrections_blocked']}")
        print(f"Visual cue used: {info['visual_used']}")
        print(f"Visual cue failed/fallback: {info['visual_failed']}")

    os.makedirs("results", exist_ok=True)

    csv_path = "results/visual_odometry_ekf_results.csv"

    fieldnames = [
        "experiment",
        "ATE_m",
        "RPE_m",
        "Mean_m",
        "Median_m",
        "Max_m",
        "P95_m",
        "corrections_used",
        "corrections_blocked",
        "visual_used",
        "visual_failed",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    plot_trajectories(
        gt_xy=gt_xy,
        trajectories=trajectories,
        save_path="results/visual_odometry_ekf_trajectories.png",
    )

    plot_ate_bar(
        results=results,
        save_path="results/visual_odometry_ekf_ate_bar.png",
    )

    print("\nSaved:")
    print(csv_path)
    print("results/visual_odometry_ekf_trajectories.png")
    print("results/visual_odometry_ekf_ate_bar.png")


if __name__ == "__main__":
    main()