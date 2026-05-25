import numpy as np


def generate_sequence(kind="urban", dur=120.0, seed=42):
    np.random.seed(seed)
    dt = 0.01
    t = np.arange(0, dur, dt)
    N = len(t)
    # Simple trajectory (figure‑eight + straight)
    x = 30 * np.sin(2 * np.pi * 0.1 * t) + 5 * np.sin(2 * np.pi * 0.03 * t)
    y = 20 * np.cos(2 * np.pi * 0.08 * t) + 10 * (10 * t / dur)
    z = 0.1 * np.sin(2 * np.pi * 0.2 * t)
    pos = np.column_stack([x, y, z])
    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    yaw = np.arctan2(vel[:, 1], vel[:, 0])
    imu_acc = acc + np.random.normal(0, 0.1, acc.shape)
    imu_gyro = np.gradient(np.column_stack([np.zeros(N), np.zeros(N), yaw]), dt, axis=0) + np.random.normal(0, 0.005,
                                                                                                            (N, 3))
    lidar_pts = [np.random.randn(1024, 3).astype(np.float32) for _ in range(N)]
    cam_imgs = [np.random.rand(128, 416, 3).astype(np.float32) for _ in range(N)]
    speed = np.linalg.norm(vel[:, :2], axis=1)
    targets = np.column_stack([pos[:, 0], pos[:, 1], speed, yaw])

    # GNSS positions (downsampled and with noise)
    gnss_step = int(1 / (10 * dt))  # 10 Hz GNSS
    gnss_idx = np.arange(0, N, gnss_step)
    gnss_pos = pos[gnss_idx] + np.random.normal(0, 1.0, (len(gnss_idx), 3))
    gnss_valid = np.random.rand(len(gnss_idx)) > 0.2  # 20% dropout

    return {
        'timestamps': t,
        'gt_pos': pos,
        'gt_vel': vel,
        'gt_att': np.column_stack([np.zeros(N), np.zeros(N), yaw]),
        'imu_acc': imu_acc,
        'imu_gyro': imu_gyro,
        'gnss_pos': gnss_pos,
        'gnss_valid': gnss_valid,
        'gnss_idx': gnss_idx,
        'lidar_pts': lidar_pts,
        'cam_imgs': cam_imgs,
        'targets': targets,
        'dt': dt
    }