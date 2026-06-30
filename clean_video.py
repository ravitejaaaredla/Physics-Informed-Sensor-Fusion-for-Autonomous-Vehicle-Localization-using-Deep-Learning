"""
clean_research_thesis_video.py

Clean research-style video generator for:
Physics-Informed Sensor Fusion for Autonomous Vehicle Localization

Run:
    python clean_research_thesis_video.py

Required project structure:
    config.py
    src/models.py
    src/raw_kitti_utils.py
    models/pinn_delta_physics.pth

Output:
    results/clean_research_thesis_demo.mp4
    results/clean_research_preview.png
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from config import Config
from src.models import FusionModel
from src.raw_kitti_utils import load_raw_sequence


# ============================================================
# Basic math and metrics
# ============================================================

def wrap_angle(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def compute_ate(pred_xy, gt_xy):
    n = min(len(pred_xy), len(gt_xy))
    pred_xy = pred_xy[:n]
    gt_xy = gt_xy[:n]
    error = pred_xy - gt_xy
    return float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))


def compute_rpe(pred_xy, gt_xy):
    n = min(len(pred_xy), len(gt_xy))

    if n < 2:
        return float("nan")

    pred_rel = pred_xy[1:n] - pred_xy[: n - 1]
    gt_rel = gt_xy[1:n] - gt_xy[: n - 1]

    return float(np.sqrt(np.mean(np.sum((pred_rel - gt_rel) ** 2, axis=1))))


def compute_current_error(pred_xy, gt_xy, idx):
    idx = min(idx, len(pred_xy) - 1, len(gt_xy) - 1)
    return float(np.linalg.norm(pred_xy[idx] - gt_xy[idx]))


def predict_delta(model, imu_win, lidar, cam, device):
    """
    Deep learning motion prediction.

    Input:
        IMU window
        LiDAR point cloud
        Camera image

    Output:
        dx_body, dy_body, dv, dyaw
    """

    model.eval()

    imu_t = torch.tensor(
        imu_win,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    lidar_t = torch.tensor(
        lidar,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    cam_t = (
        torch.tensor(cam, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        delta = model(imu_t, lidar_t, cam_t).cpu().numpy()[0]

    return delta.astype(np.float32)


# ============================================================
# Lightweight LiDAR cue
# ============================================================

def filter_lidar_points(points):
    pts = np.asarray(points, dtype=np.float32)

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    dist = np.linalg.norm(pts[:, :3], axis=1)

    mask = (
        (dist > 2.0)
        & (dist < 45.0)
        & (z > -2.2)
        & (z < 2.0)
        & (x > -30.0)
        & (x < 30.0)
        & (y > -30.0)
        & (y < 50.0)
    )

    filtered = pts[mask]

    if len(filtered) < 50:
        return pts

    return filtered


def robust_lidar_centroid(points):
    pts = filter_lidar_points(points)

    med = np.median(pts[:, :3], axis=0)
    dist = np.linalg.norm(pts[:, :3] - med, axis=1)

    keep = dist < np.percentile(dist, 70)
    pts_keep = pts[keep]

    if len(pts_keep) < 30:
        pts_keep = pts

    return np.mean(pts_keep[:, :3], axis=0)


def estimate_lidar_motion(prev_points, curr_points):
    """
    Lightweight LiDAR relative-motion cue.

    This is NOT full ICP, NDT, or LOAM odometry.
    It estimates motion from robust scene centroid shift.
    """

    c_prev = robust_lidar_centroid(prev_points)
    c_curr = robust_lidar_centroid(curr_points)

    scene_shift = c_curr - c_prev

    dx_body = -float(scene_shift[0])
    dy_body = -float(scene_shift[1])

    step = np.sqrt(dx_body ** 2 + dy_body ** 2)

    if step > 5.0:
        return 0.0, 0.0, 0.0

    quality = float(np.exp(-step / 5.0))

    return dx_body, dy_body, quality


# ============================================================
# Lightweight visual cue
# ============================================================

def to_gray_uint8(img):
    img = np.asarray(img)

    if img.dtype != np.uint8:
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img_u8 = img.copy()

    if img_u8.ndim == 2:
        return img_u8

    return cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)


def estimate_visual_motion(prev_img, curr_img, pinn_dx, pinn_dy):
    """
    Lightweight monocular visual motion cue.

    This is NOT full calibrated visual odometry.
    Scale is taken from the PINN-predicted displacement magnitude.
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
    matches = matches[: min(200, len(matches))]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Approximate intrinsics only for visual cue direction.
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

    E, _ = cv2.findEssentialMat(
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
        _, _, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K)
    except cv2.error:
        return 0.0, 0.0, 0.0

    if pose_mask is None:
        return 0.0, 0.0, 0.0

    inliers = int(np.sum(pose_mask > 0))

    if inliers < 25:
        return 0.0, 0.0, 0.0

    t = t.reshape(3)
    t = t / (np.linalg.norm(t) + 1e-8)

    direction_dx = float(t[2])
    direction_dy = float(-t[0])

    direction_norm = np.sqrt(direction_dx ** 2 + direction_dy ** 2) + 1e-8
    direction_dx /= direction_norm
    direction_dy /= direction_norm

    pinn_step = np.sqrt(pinn_dx ** 2 + pinn_dy ** 2)

    if pinn_step <= 1e-6 or pinn_step > 5.0:
        return 0.0, 0.0, 0.0

    dx_body = direction_dx * pinn_step
    dy_body = direction_dy * pinn_step

    quality = min(1.0, inliers / 120.0)

    return dx_body, dy_body, quality


# ============================================================
# EKF Localization
# ============================================================

class LocalizationEKF:
    def __init__(self, config):
        self.config = config
        self.dt = config.dt

        # State = [x, y, velocity, yaw]
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

        self.R = np.diag(
            [
                gnss_noise ** 2,
                gnss_noise ** 2,
            ]
        ).astype(np.float32)

    def initialize(self, initial_state):
        self.x[:] = initial_state.astype(np.float32)

    def predict(self, dx_body, dy_body, dv, gyro_z):
        px, py, v, yaw = self.x

        # IMU yaw-rate integration
        dyaw = gyro_z * self.dt

        c = np.cos(yaw)
        s = np.sin(yaw)

        # Convert local body motion to global/world motion
        dx_world = c * dx_body - s * dy_body
        dy_world = s * dx_body + c * dy_body

        px_new = px + dx_world
        py_new = py + dy_world
        v_new = max(0.0, v + dv)
        yaw_new = wrap_angle(yaw + dyaw)

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

    def update_position(self, z_xy):
        z = np.asarray(z_xy, dtype=np.float32)

        H = np.zeros((2, 4), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        residual = z - H @ self.x

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ residual
        self.x[3] = wrap_angle(self.x[3])

        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ H) @ self.P


# ============================================================
# Run localization methods
# ============================================================

def run_method(
    method_name,
    model,
    states,
    lidar_pts,
    cam_imgs,
    imu_data,
    config,
    device,
):
    """
    method_name:
        base   = PINN translation + IMU yaw + EKF/OXTS
        lidar  = base + lightweight LiDAR cue
        visual = base + lightweight visual cue
    """

    ekf = LocalizationEKF(config)

    start_frame = config.window_size - 1
    ekf.initialize(states[start_frame])

    pred_xy = [ekf.x[:2].copy()]
    frame_indices = [start_frame]

    correction_interval = 10

    lidar_weight = 0.15
    visual_weight = 0.10

    cue_used = 0
    cue_failed = 0

    for end in range(start_frame, len(states) - 1):
        start = end - config.window_size + 1

        imu_win = imu_data[start:end + 1]
        lidar = lidar_pts[end]
        cam = cam_imgs[end]

        delta = predict_delta(
            model=model,
            imu_win=imu_win,
            lidar=lidar,
            cam=cam,
            device=device,
        )

        pinn_dx = float(delta[0])
        pinn_dy = float(delta[1])
        pinn_dv = float(delta[2])

        dx_body = pinn_dx
        dy_body = pinn_dy

        if method_name == "lidar" and end > 0:
            lidar_dx, lidar_dy, quality = estimate_lidar_motion(
                lidar_pts[end - 1],
                lidar,
            )

            if quality > 0.20:
                w = lidar_weight * quality
                dx_body = (1.0 - w) * pinn_dx + w * lidar_dx
                dy_body = (1.0 - w) * pinn_dy + w * lidar_dy
                cue_used += 1
            else:
                cue_failed += 1

        elif method_name == "visual" and end > 0:
            visual_dx, visual_dy, quality = estimate_visual_motion(
                prev_img=cam_imgs[end - 1],
                curr_img=cam,
                pinn_dx=pinn_dx,
                pinn_dy=pinn_dy,
            )

            if quality > 0.20:
                w = visual_weight * quality
                dx_body = (1.0 - w) * pinn_dx + w * visual_dx
                dy_body = (1.0 - w) * pinn_dy + w * visual_dy
                cue_used += 1
            else:
                cue_failed += 1

        gyro_z = float(imu_win[-1, 5])

        ekf.predict(
            dx_body=dx_body,
            dy_body=dy_body,
            dv=pinn_dv,
            gyro_z=gyro_z,
        )

        corrected_frame = end + 1

        if corrected_frame % correction_interval == 0:
            ekf.update_position(states[corrected_frame, :2])

        pred_xy.append(ekf.x[:2].copy())
        frame_indices.append(corrected_frame)

    return (
        np.asarray(pred_xy, dtype=np.float32),
        np.asarray(frame_indices, dtype=np.int32),
        cue_used,
        cue_failed,
    )


# ============================================================
# Clean drawing style
# ============================================================

BG = (246, 248, 250)
PANEL = (255, 255, 255)
TEXT = (31, 41, 55)
MUTED = (107, 114, 128)
GRID = (225, 229, 235)

BLACK = (20, 20, 20)
BASE = (40, 95, 210)
LIDAR = (215, 105, 40)
VISUAL = (40, 150, 80)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_text(img, text, org, size=0.7, color=TEXT, thickness=1):
    cv2.putText(
        img,
        text,
        org,
        FONT,
        size,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_card(canvas, x, y, w, h, title):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (218, 224, 230), 1)

    cv2.rectangle(canvas, (x, y), (x + w, y + 48), (238, 242, 246), -1)
    cv2.line(canvas, (x, y + 48), (x + w, y + 48), (218, 224, 230), 1)

    put_text(canvas, title, (x + 18, y + 32), 0.72, TEXT, 2)


def fit_image_keep_aspect(img, target_w, target_h):
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2

    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    return canvas


def draw_camera_content(img, w, h):
    panel = np.ones((h, w, 3), dtype=np.uint8) * 255

    if img.dtype != np.uint8:
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img_u8 = img.copy()

    if img_u8.ndim == 2:
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

    fitted = fit_image_keep_aspect(img_bgr, w - 24, h - 68)

    panel[56:56 + fitted.shape[0], 12:12 + fitted.shape[1]] = fitted

    return panel


def draw_lidar_content(points, w, h):
    panel = np.ones((h, w, 3), dtype=np.uint8) * 255

    plot_x = 18
    plot_y = 62
    plot_w = w - 36
    plot_h = h - 86

    cv2.rectangle(
        panel,
        (plot_x, plot_y),
        (plot_x + plot_w, plot_y + plot_h),
        (250, 250, 250),
        -1,
    )

    pts = np.asarray(points, dtype=np.float32)

    if len(pts) > 0:
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        mask = (
            (x > 0.0)
            & (x < 50.0)
            & (y > -25.0)
            & (y < 25.0)
            & (z > -2.5)
            & (z < 2.0)
        )

        pts = pts[mask]

        if len(pts) > 0:
            x = pts[:, 0]
            y = pts[:, 1]
            z = pts[:, 2]

            px = plot_x + ((y + 25.0) / 50.0 * (plot_w - 1)).astype(np.int32)
            py = plot_y + ((50.0 - x) / 50.0 * (plot_h - 1)).astype(np.int32)

            z_norm = np.clip((z + 2.5) / 4.5, 0.0, 1.0)
            colors = (180 * z_norm + 50).astype(np.uint8)

            for i in range(len(px)):
                cv2.circle(
                    panel,
                    (int(px[i]), int(py[i])),
                    1,
                    (int(255 - colors[i]), int(colors[i]), 80),
                    -1,
                )

    for distance in [10, 20, 30, 40, 50]:
        gy = plot_y + int((50.0 - distance) / 50.0 * (plot_h - 1))

        cv2.line(
            panel,
            (plot_x, gy),
            (plot_x + plot_w, gy),
            GRID,
            1,
        )

        put_text(
            panel,
            f"{distance} m",
            (plot_x + 6, gy - 4),
            0.42,
            MUTED,
            1,
        )

    ego_x = plot_x + plot_w // 2
    ego_y = plot_y + plot_h - 16

    cv2.circle(panel, (ego_x, ego_y), 7, (30, 60, 220), -1)
    cv2.line(panel, (ego_x, ego_y), (ego_x, ego_y - 35), (30, 60, 220), 3)

    put_text(panel, "Forward", (ego_x + 12, ego_y - 30), 0.45, MUTED, 1)

    return panel


def xy_to_pixel(xy, bounds, plot):
    xmin, xmax, ymin, ymax = bounds
    x0, y0, w, h = plot

    px = x0 + (xy[0] - xmin) / (xmax - xmin + 1e-8) * w
    py = y0 + (ymax - xy[1]) / (ymax - ymin + 1e-8) * h

    return int(px), int(py)


def draw_polyline(panel, xy, bounds, plot, color, thickness=3):
    if len(xy) < 2:
        return

    for i in range(1, len(xy)):
        p1 = xy_to_pixel(xy[i - 1], bounds, plot)
        p2 = xy_to_pixel(xy[i], bounds, plot)

        cv2.line(
            panel,
            p1,
            p2,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_trajectory_content(gt_xy, predictions, frame_idx, w, h):
    panel = np.ones((h, w, 3), dtype=np.uint8) * 255

    plot = (70, 72, w - 110, h - 150)
    x0, y0, pw, ph = plot

    all_xy = [gt_xy]

    for pred in predictions.values():
        all_xy.append(pred)

    all_xy = np.vstack(all_xy)

    pad = 8.0

    xmin = float(np.min(all_xy[:, 0]) - pad)
    xmax = float(np.max(all_xy[:, 0]) + pad)
    ymin = float(np.min(all_xy[:, 1]) - pad)
    ymax = float(np.max(all_xy[:, 1]) + pad)

    bounds = (xmin, xmax, ymin, ymax)

    cv2.rectangle(
        panel,
        (x0, y0),
        (x0 + pw, y0 + ph),
        (250, 251, 253),
        -1,
    )

    cv2.rectangle(
        panel,
        (x0, y0),
        (x0 + pw, y0 + ph),
        (210, 210, 210),
        1,
    )

    for k in range(1, 5):
        gx = x0 + int(k * pw / 5)
        gy = y0 + int(k * ph / 5)

        cv2.line(panel, (gx, y0), (gx, y0 + ph), GRID, 1)
        cv2.line(panel, (x0, gy), (x0 + pw, gy), GRID, 1)

    n = min(frame_idx + 1, len(gt_xy))

    draw_polyline(panel, gt_xy[:n], bounds, plot, BLACK, 3)
    draw_polyline(panel, predictions["Base PINN+IMU+EKF"][:n], bounds, plot, BASE, 3)
    draw_polyline(panel, predictions["LiDAR cue"][:n], bounds, plot, LIDAR, 3)
    draw_polyline(panel, predictions["Visual cue"][:n], bounds, plot, VISUAL, 3)

    if n > 0:
        cv2.circle(
            panel,
            xy_to_pixel(gt_xy[n - 1], bounds, plot),
            6,
            BLACK,
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            panel,
            xy_to_pixel(predictions["Base PINN+IMU+EKF"][n - 1], bounds, plot),
            6,
            BASE,
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            panel,
            xy_to_pixel(predictions["LiDAR cue"][n - 1], bounds, plot),
            6,
            LIDAR,
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            panel,
            xy_to_pixel(predictions["Visual cue"][n - 1], bounds, plot),
            6,
            VISUAL,
            -1,
            cv2.LINE_AA,
        )

    put_text(panel, "X position (m)", (x0 + pw // 2 - 60, h - 28), 0.55, MUTED, 1)
    put_text(panel, "Y", (22, y0 + ph // 2), 0.55, MUTED, 1)

    legend_y = h - 82
    legend_x = 72

    legend_items = [
        ("OXTS ground truth", BLACK),
        ("Base PINN+IMU+EKF", BASE),
        ("LiDAR cue", LIDAR),
        ("Visual cue", VISUAL),
    ]

    for label, color in legend_items:
        cv2.line(
            panel,
            (legend_x, legend_y),
            (legend_x + 36, legend_y),
            color,
            4,
            cv2.LINE_AA,
        )

        put_text(
            panel,
            label,
            (legend_x + 46, legend_y + 6),
            0.50,
            TEXT,
            1,
        )

        legend_x += 250

    return panel


def draw_metric_row(panel, name, color, ate, rpe, current_error, x, y, w):
    put_text(panel, name, (x, y), 0.62, color, 2)

    put_text(panel, f"ATE  {ate:6.3f} m", (x, y + 36), 0.56, TEXT, 1)
    put_text(panel, f"RPE  {rpe:6.3f} m", (x, y + 66), 0.56, TEXT, 1)

    put_text(
        panel,
        f"Current error  {current_error:5.2f} m",
        (x, y + 96),
        0.52,
        MUTED,
        1,
    )

    bar_x = x
    bar_y = y + 118
    bar_w = w - 40
    bar_h = 14

    cv2.rectangle(
        panel,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + bar_h),
        (230, 232, 236),
        -1,
    )

    filled = int(np.clip(current_error / 20.0, 0.0, 1.0) * bar_w)

    cv2.rectangle(
        panel,
        (bar_x, bar_y),
        (bar_x + filled, bar_y + bar_h),
        color,
        -1,
    )


def draw_metrics_content(gt_xy, predictions, metrics, frame_idx, w, h):
    panel = np.ones((h, w, 3), dtype=np.uint8) * 255

    put_text(panel, "Current Evaluation", (22, 45), 0.82, TEXT, 2)

    y = 95
    row_h = 160

    for name, color in [
        ("Base PINN+IMU+EKF", BASE),
        ("LiDAR cue", LIDAR),
        ("Visual cue", VISUAL),
    ]:
        current_error = compute_current_error(
            predictions[name],
            gt_xy,
            frame_idx,
        )

        draw_metric_row(
            panel=panel,
            name=name,
            color=color,
            ate=metrics[name]["ATE"],
            rpe=metrics[name]["RPE"],
            current_error=current_error,
            x=24,
            y=y,
            w=w,
        )

        y += row_h

    box_y = h - 145

    cv2.rectangle(
        panel,
        (20, box_y),
        (w - 20, h - 24),
        (244, 247, 250),
        -1,
    )

    cv2.rectangle(
        panel,
        (20, box_y),
        (w - 20, h - 24),
        (222, 226, 232),
        1,
    )

    put_text(panel, "Interpretation", (36, box_y + 34), 0.60, TEXT, 2)
    put_text(panel, "Lower ATE means better global", (36, box_y + 67), 0.48, MUTED, 1)
    put_text(panel, "trajectory accuracy.", (36, box_y + 91), 0.48, MUTED, 1)

    return panel


def build_video_frame(cam, lidar, gt_xy, predictions, metrics, i, n):
    """
    Build one clean 1920 x 1080 video frame.
    """

    W = 1920
    H = 1080

    frame = np.ones((H, W, 3), dtype=np.uint8)
    frame[:] = BG

    # Header
    cv2.rectangle(frame, (0, 0), (W, 92), (17, 24, 39), -1)

    put_text(
        frame,
        "Physics-Informed Multi-Sensor Localization",
        (42, 38),
        1.05,
        (255, 255, 255),
        2,
    )

    put_text(
        frame,
        "Camera + LiDAR + IMU  |  PINN motion prediction + IMU yaw propagation + EKF/OXTS correction",
        (42, 72),
        0.62,
        (210, 218, 230),
        1,
    )

    put_text(
        frame,
        f"Frame {i + 1}/{n}",
        (1740, 57),
        0.62,
        (230, 234, 240),
        1,
    )

    margin = 28
    gap = 22

    top_y = 112
    top_h = 390

    cam_x = margin
    cam_w = 900

    lidar_x = cam_x + cam_w + gap
    lidar_w = 460

    metrics_x = lidar_x + lidar_w + gap
    metrics_w = W - metrics_x - margin

    bottom_y = top_y + top_h + gap
    bottom_h = H - bottom_y - margin

    traj_x = margin
    traj_w = metrics_x - margin - gap

    # Camera card
    draw_card(frame, cam_x, top_y, cam_w, top_h, "Front camera")
    cam_content = draw_camera_content(cam, cam_w, top_h)
    frame[top_y + 49:top_y + top_h, cam_x:cam_x + cam_w] = cam_content[49:top_h, :]

    # LiDAR card
    draw_card(frame, lidar_x, top_y, lidar_w, top_h, "LiDAR bird's-eye view")
    lidar_content = draw_lidar_content(lidar, lidar_w, top_h)
    frame[top_y + 49:top_y + top_h, lidar_x:lidar_x + lidar_w] = lidar_content[49:top_h, :]

    # Metrics card
    metrics_h = H - top_y - margin
    draw_card(frame, metrics_x, top_y, metrics_w, metrics_h, "Metrics")

    metrics_content = draw_metrics_content(
        gt_xy=gt_xy,
        predictions=predictions,
        metrics=metrics,
        frame_idx=i,
        w=metrics_w,
        h=metrics_h,
    )

    frame[top_y + 49:H - margin, metrics_x:metrics_x + metrics_w] = metrics_content[49:metrics_h, :]

    # Trajectory card
    draw_card(frame, traj_x, bottom_y, traj_w, bottom_h, "Time-aligned trajectory")

    traj_content = draw_trajectory_content(
        gt_xy=gt_xy,
        predictions=predictions,
        frame_idx=i,
        w=traj_w,
        h=bottom_h,
    )

    frame[bottom_y + 49:bottom_y + bottom_h, traj_x:traj_x + traj_w] = traj_content[49:bottom_h, :]

    return frame


def make_video(cam_imgs, lidar_pts, gt_xy, predictions, metrics, output_path, fps=10):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    W = 1920
    H = 1080

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    n = min(len(cam_imgs), len(lidar_pts), len(gt_xy))

    for pred in predictions.values():
        n = min(n, len(pred))

    preview_frame = None

    for i in range(n):
        frame = build_video_frame(
            cam=cam_imgs[i],
            lidar=lidar_pts[i],
            gt_xy=gt_xy,
            predictions=predictions,
            metrics=metrics,
            i=i,
            n=n,
        )

        writer.write(frame)

        if i == n // 2:
            preview_frame = frame.copy()

    writer.release()

    if preview_frame is not None:
        preview_path = output_path.parent / "clean_research_preview.png"
        cv2.imwrite(str(preview_path), preview_frame)
        print(f"Saved preview image: {preview_path}")

    print(f"Saved video: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model_path = Path("models/pinn_delta_physics.pth")

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. First run: python train_physics_loss.py"
        )

    model = FusionModel(config).to(device)
    model.load_state_dict(
        torch.load(str(model_path), map_location=device)
    )
    model.eval()

    print("Loading KITTI Raw sequence...")
    states, lidar_pts, cam_imgs, imu_data = load_raw_sequence(
        config.seq_dir,
        config,
    )

    print("Running Base PINN + IMU yaw + EKF/OXTS...")
    base_xy, base_idx, _, _ = run_method(
        method_name="base",
        model=model,
        states=states,
        lidar_pts=lidar_pts,
        cam_imgs=cam_imgs,
        imu_data=imu_data,
        config=config,
        device=device,
    )

    print("Running LiDAR cue method...")
    lidar_xy, lidar_idx, lidar_cue_used, lidar_cue_failed = run_method(
        method_name="lidar",
        model=model,
        states=states,
        lidar_pts=lidar_pts,
        cam_imgs=cam_imgs,
        imu_data=imu_data,
        config=config,
        device=device,
    )

    print("Running Visual cue method...")
    visual_xy, visual_idx, visual_cue_used, visual_cue_failed = run_method(
        method_name="visual",
        model=model,
        states=states,
        lidar_pts=lidar_pts,
        cam_imgs=cam_imgs,
        imu_data=imu_data,
        config=config,
        device=device,
    )

    n = min(
        len(base_xy),
        len(lidar_xy),
        len(visual_xy),
        len(base_idx),
        len(lidar_idx),
        len(visual_idx),
    )

    frame_indices = base_idx[:n].astype(int)

    gt_xy = np.asarray(
        [states[int(idx), :2] for idx in frame_indices],
        dtype=np.float32,
    )

    cam_used = [
        cam_imgs[int(idx)]
        for idx in frame_indices
    ]

    lidar_used_frames = [
        lidar_pts[int(idx)]
        for idx in frame_indices
    ]

    predictions = {
        "Base PINN+IMU+EKF": base_xy[:n],
        "LiDAR cue": lidar_xy[:n],
        "Visual cue": visual_xy[:n],
    }

    metrics = {}

    for name, pred in predictions.items():
        metrics[name] = {
            "ATE": compute_ate(pred, gt_xy),
            "RPE": compute_rpe(pred, gt_xy),
        }

    print("\nFinal metrics")
    print("=" * 70)

    for name, vals in metrics.items():
        print(
            f"{name:22s} | ATE: {vals['ATE']:.3f} m | RPE: {vals['RPE']:.3f} m"
        )

    print("\nCue usage")
    print("=" * 70)
    print(f"LiDAR cue used: {lidar_cue_used}, fallback: {lidar_cue_failed}")
    print(f"Visual cue used: {visual_cue_used}, fallback: {visual_cue_failed}")

    output_path = "results/clean_research_thesis_demo.mp4"

    print("\nCreating clean research-style video...")

    make_video(
        cam_imgs=cam_used,
        lidar_pts=lidar_used_frames,
        gt_xy=gt_xy,
        predictions=predictions,
        metrics=metrics,
        output_path=output_path,
        fps=10,
    )


if __name__ == "__main__":
    main()