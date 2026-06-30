
import os
import cv2
import numpy as np
import torch

from config import Config
from src.models import FusionModel
from src.raw_kitti_utils import load_raw_sequence


# ============================================================
# Utilities
# ============================================================

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


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

    pred_xy = pred_xy[:n]
    gt_xy = gt_xy[:n]

    pred_rel = pred_xy[1:] - pred_xy[:-1]
    gt_rel = gt_xy[1:] - gt_xy[:-1]

    return float(np.sqrt(np.mean(np.sum((pred_rel - gt_rel) ** 2, axis=1))))


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


# ============================================================
# Visual motion cue
# ============================================================

def to_gray_uint8(img):
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)


def estimate_visual_motion(prev_img, curr_img, pinn_dx, pinn_dy):
    """
    Lightweight visual motion cue.

    This is not full calibrated visual odometry.
    It estimates visual motion direction using ORB + essential matrix.
    Metric scale is taken from the PINN-predicted displacement magnitude.
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

    # Approximate intrinsics for resized frames.
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
    t = t / (np.linalg.norm(t) + 1e-8)

    # Camera coordinates:
    # z = forward, x = right.
    # Vehicle/body coordinates:
    # x_body = forward, y_body = left.
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
# EKF
# ============================================================

class LocalizationEKF:
    def __init__(self, config):
        self.config = config
        self.dt = config.dt

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

        dyaw = gyro_z * self.dt

        c = np.cos(yaw)
        s = np.sin(yaw)

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
# Time-aligned visual-cue localization
# ============================================================

def run_visual_cue_localization(
    model,
    states,
    lidar_pts,
    cam_imgs,
    imu_data,
    config,
    device,
):
    """
    Runs the best normal localization method used for the clean video:

        Visual cue + PINN translation + IMU yaw + EKF/OXTS correction

    The output is time-aligned:
    each camera frame, LiDAR scan, OXTS point, and predicted point
    correspond to the same frame index.
    """

    ekf = LocalizationEKF(config)

    start_frame = config.window_size - 1
    ekf.initialize(states[start_frame])

    pred_xy = [ekf.x[:2].copy()]
    frame_indices = [start_frame]

    correction_interval = 10
    visual_weight = 0.10

    visual_used = 0
    visual_failed = 0

    for end in range(start_frame, len(states) - 1):
        start = end - config.window_size + 1
        corrected_frame = end + 1

        imu_win = imu_data[start:end + 1]

        lidar_for_model = lidar_pts[end]
        cam_for_model = cam_imgs[end]

        delta = predict_delta(
            model=model,
            imu_win=imu_win,
            lidar=lidar_for_model,
            cam=cam_for_model,
            device=device,
        )

        pinn_dx = float(delta[0])
        pinn_dy = float(delta[1])
        pinn_dv = float(delta[2])

        dx_body = pinn_dx
        dy_body = pinn_dy

        # Visual motion from current frame to next frame.
        prev_cam = cam_imgs[end]
        curr_cam = cam_imgs[corrected_frame]

        visual_dx, visual_dy, quality = estimate_visual_motion(
            prev_img=prev_cam,
            curr_img=curr_cam,
            pinn_dx=pinn_dx,
            pinn_dy=pinn_dy,
        )

        if quality > 0.20:
            w = visual_weight * quality
            dx_body = (1.0 - w) * pinn_dx + w * visual_dx
            dy_body = (1.0 - w) * pinn_dy + w * visual_dy
            visual_used += 1
        else:
            visual_failed += 1

        gyro_z = float(imu_win[-1, 5])

        ekf.predict(
            dx_body=dx_body,
            dy_body=dy_body,
            dv=pinn_dv,
            gyro_z=gyro_z,
        )

        if corrected_frame % correction_interval == 0:
            ekf.update_position(states[corrected_frame, :2])

        pred_xy.append(ekf.x[:2].copy())
        frame_indices.append(corrected_frame)

    return (
        np.asarray(pred_xy, dtype=np.float32),
        np.asarray(frame_indices, dtype=np.int32),
        visual_used,
        visual_failed,
    )


# ============================================================
# Drawing functions
# ============================================================

def draw_small_label(img, text, x, y, color=(255, 255, 255)):
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_camera_panel(img, width=640, height=360):
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (width, height))

    # Small label only.
    cv2.rectangle(img_bgr, (10, 10), (95, 35), (0, 0, 0), -1)
    draw_small_label(img_bgr, "Camera", 18, 29)

    return img_bgr


def draw_lidar_bev(points, width=640, height=360):
    """
    Simple LiDAR BEV:
    x forward, y left, z up.
    """

    canvas = np.ones((height, width, 3), dtype=np.uint8) * 250

    pts = np.asarray(points, dtype=np.float32)

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

        px = ((y + 25.0) / 50.0 * (width - 1)).astype(np.int32)
        py = ((50.0 - x) / 50.0 * (height - 1)).astype(np.int32)

        z_norm = np.clip((z + 2.5) / 4.5, 0.0, 1.0)
        color_value = (255 * z_norm).astype(np.uint8)

        for i in range(len(px)):
            cv2.circle(
                canvas,
                (int(px[i]), int(py[i])),
                1,
                (30, int(color_value[i]), int(255 - color_value[i])),
                -1,
            )

    # Light grid lines
    for m in [10, 20, 30, 40, 50]:
        gy = int((50.0 - m) / 50.0 * (height - 1))

        cv2.line(
            canvas,
            (0, gy),
            (width, gy),
            (225, 225, 225),
            1,
        )

        cv2.putText(
            canvas,
            f"{m}m",
            (8, max(18, gy - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )

    # Ego vehicle marker
    ego_x = width // 2
    ego_y = height - 18

    cv2.circle(canvas, (ego_x, ego_y), 5, (0, 0, 200), -1)
    cv2.line(canvas, (ego_x, ego_y), (ego_x, ego_y - 24), (0, 0, 200), 2)

    cv2.rectangle(canvas, (10, 10), (112, 35), (255, 255, 255), -1)
    cv2.putText(
        canvas,
        "LiDAR BEV",
        (18, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return canvas


def normalize_xy_to_canvas(xy, xmin, xmax, ymin, ymax, width, height, margin=45):
    x = xy[0]
    y = xy[1]

    px = margin + (x - xmin) / (xmax - xmin + 1e-8) * (width - 2 * margin)
    py = margin + (ymax - y) / (ymax - ymin + 1e-8) * (height - 2 * margin)

    return int(px), int(py)


def draw_trajectory_panel(gt_xy, pred_xy, frame_idx, ate, rpe, width=1280, height=360):
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 252

    n = min(frame_idx + 1, len(gt_xy), len(pred_xy))

    all_xy = np.vstack([gt_xy, pred_xy])

    xmin = np.min(all_xy[:, 0])
    xmax = np.max(all_xy[:, 0])
    ymin = np.min(all_xy[:, 1])
    ymax = np.max(all_xy[:, 1])

    gt_part = gt_xy[:n]
    pred_part = pred_xy[:n]

    # Ground truth: black
    for i in range(1, len(gt_part)):
        p1 = normalize_xy_to_canvas(gt_part[i - 1], xmin, xmax, ymin, ymax, width, height)
        p2 = normalize_xy_to_canvas(gt_part[i], xmin, xmax, ymin, ymax, width, height)
        cv2.line(canvas, p1, p2, (0, 0, 0), 2, cv2.LINE_AA)

    # Estimated: red
    for i in range(1, len(pred_part)):
        p1 = normalize_xy_to_canvas(pred_part[i - 1], xmin, xmax, ymin, ymax, width, height)
        p2 = normalize_xy_to_canvas(pred_part[i], xmin, xmax, ymin, ymax, width, height)
        cv2.line(canvas, p1, p2, (0, 0, 220), 2, cv2.LINE_AA)

    gt_p = normalize_xy_to_canvas(gt_part[-1], xmin, xmax, ymin, ymax, width, height)
    pr_p = normalize_xy_to_canvas(pred_part[-1], xmin, xmax, ymin, ymax, width, height)

    cv2.circle(canvas, gt_p, 5, (0, 0, 0), -1)
    cv2.circle(canvas, pr_p, 5, (0, 0, 220), -1)

    current_error = float(np.linalg.norm(gt_part[-1] - pred_part[-1]))

    # Minimal text
    cv2.putText(
        canvas,
        "Localization",
        (25, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.line(canvas, (25, height - 38), (65, height - 38), (0, 0, 0), 2)
    cv2.putText(
        canvas,
        "Ground Truth",
        (75, height - 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.line(canvas, (220, height - 38), (260, height - 38), (0, 0, 220), 2)
    cv2.putText(
        canvas,
        "Estimated",
        (270, height - 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 180),
        1,
        cv2.LINE_AA,
    )

    text = f"Error: {current_error:.2f} m   ATE: {ate:.2f} m   RPE: {rpe:.2f} m"
    cv2.putText(
        canvas,
        text,
        (width - 440, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )

    return canvas


def make_video(cam_imgs, lidar_pts, gt_xy, pred_xy, output_path, ate, rpe, fps=10):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    out_w = 1280
    out_h = 720

    top_h = 360
    bottom_h = 360

    panel_w = 640

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        raise RuntimeError("Could not open video writer. Try changing codec or output path.")

    n = min(len(cam_imgs), len(lidar_pts), len(gt_xy), len(pred_xy))

    for i in range(n):
        frame = np.ones((out_h, out_w, 3), dtype=np.uint8) * 245

        cam_panel = draw_camera_panel(
            cam_imgs[i],
            width=panel_w,
            height=top_h,
        )

        bev_panel = draw_lidar_bev(
            lidar_pts[i],
            width=panel_w,
            height=top_h,
        )

        traj_panel = draw_trajectory_panel(
            gt_xy=gt_xy,
            pred_xy=pred_xy,
            frame_idx=i,
            ate=ate,
            rpe=rpe,
            width=out_w,
            height=bottom_h,
        )

        frame[0:top_h, 0:panel_w] = cam_panel
        frame[0:top_h, panel_w:out_w] = bev_panel
        frame[top_h:out_h, 0:out_w] = traj_panel

        writer.write(frame)

    writer.release()


# ============================================================
# Main
# ============================================================

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

    print("\nRunning research-style localization video method...")
    pred_xy, frame_indices, visual_used, visual_failed = run_visual_cue_localization(
        model=model,
        states=states,
        lidar_pts=lidar_pts,
        cam_imgs=cam_imgs,
        imu_data=imu_data,
        config=config,
        device=device,
    )

    frame_indices = frame_indices.astype(int)

    gt_xy = np.asarray(
        [states[int(i), :2] for i in frame_indices],
        dtype=np.float32,
    )

    cam_used = [
        cam_imgs[int(i)]
        for i in frame_indices
    ]

    lidar_used_frames = [
        lidar_pts[int(i)]
        for i in frame_indices
    ]

    ate = compute_ate(pred_xy, gt_xy)
    rpe = compute_rpe(pred_xy, gt_xy)

    print("\nResearch-style demo metrics")
    print("=" * 60)
    print(f"Method: Visual cue + PINN + IMU + EKF/OXTS")
    print(f"ATE: {ate:.3f} m")
    print(f"RPE: {rpe:.3f} m")
    print(f"Visual cue used: {visual_used}")
    print(f"Visual cue failed/fallback: {visual_failed}")

    output_path = "results/research_style_localization_video.mp4"

    print("\nCreating clean research-style video...")

    make_video(
        cam_imgs=cam_used,
        lidar_pts=lidar_used_frames,
        gt_xy=gt_xy,
        pred_xy=pred_xy,
        output_path=output_path,
        ate=ate,
        rpe=rpe,
        fps=10,
    )

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()