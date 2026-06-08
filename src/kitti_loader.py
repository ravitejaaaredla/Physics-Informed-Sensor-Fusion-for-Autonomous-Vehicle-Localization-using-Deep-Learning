import numpy as np
from pathlib import Path
from PIL import Image

def load_kitti_sequence(config, sequence="00"):
    """
    Loads KITTI odometry data (unified structure).
    Returns:
        bevs, imu_data, gnss_pos, cam_imgs, targets_norm, lidar_pts, pos_mean, pos_std
    """
    root = Path(config.kitti_root)   # should point to "data/kitti_unified"
    poses_file = root / "poses" / f"{sequence}.txt"
    if not poses_file.exists():
        raise FileNotFoundError(f"Poses file not found: {poses_file}")
    poses = np.loadtxt(poses_file).reshape(-1, 3, 4)
    num_frames = min(config.max_frames, len(poses))
    poses = poses[:num_frames]

    # Original targets (physical units)
    targets_orig = []
    for i in range(num_frames):
        p = poses[i, :3, 3]
        Rmat = poses[i, :3, :3]
        yaw = np.arctan2(Rmat[1,0], Rmat[0,0])
        if i == 0:
            v = 0.0
        else:
            dt = 0.1
            dp = p - poses[i-1, :3, 3]
            v = np.linalg.norm(dp) / dt
        targets_orig.append([p[0], p[1], v, yaw])
    targets_orig = np.array(targets_orig)

    # Normalise positions to zero mean, unit variance (per sequence)
    pos_mean = np.mean(targets_orig[:,:2], axis=0)
    pos_std = np.std(targets_orig[:,:2], axis=0) + 1e-8
    targets_norm = targets_orig.copy()
    targets_norm[:,:2] = (targets_orig[:,:2] - pos_mean) / pos_std

    # LiDAR
    velo_dir = root / "sequences" / sequence / "velodyne"
    lidar_pts = []
    for i in range(num_frames):
        bin_file = velo_dir / f"{i:06d}.bin"
        if bin_file.exists():
            pts = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)[:,:3]
            idx = np.random.choice(len(pts), min(1024, len(pts)), replace=False)
            pts = pts[idx]
            lidar_pts.append(pts.astype(np.float32))
        else:
            lidar_pts.append(np.random.randn(1024,3).astype(np.float32))

    # Camera
    img_dir = root / "sequences" / sequence / "image_2"
    cam_imgs = []
    for i in range(num_frames):
        img_file = img_dir / f"{i:06d}.png"
        if img_file.exists():
            img = Image.open(img_file).resize((config.camera_img_w, config.camera_img_h))
            img = np.array(img, dtype=np.float32) / 255.0
            cam_imgs.append(img)
        else:
            cam_imgs.append(np.zeros((config.camera_img_h, config.camera_img_w, 3), dtype=np.float32))

    # Synthetic IMU from ground truth (using original, non‑normalised targets)
    dt = 0.1
    imu_data = []
    for i in range(num_frames-1):
        v1 = targets_orig[i, :2]
        v2 = targets_orig[i+1, :2]
        acc = (v2 - v1) / dt
        yaw1 = targets_orig[i,3]
        yaw2 = targets_orig[i+1,3]
        yaw_rate = (yaw2 - yaw1) / dt
        imu_data.append([acc[0], acc[1], 0.0, 0.0, 0.0, yaw_rate])
    imu_data.append(imu_data[-1])
    imu_data = np.array(imu_data)

    # GNSS: downsample ground truth with noise (using original positions)
    gnss_step = int(config.imu_rate / config.gnss_rate)
    gnss_idx = np.arange(0, num_frames, gnss_step)
    gnss_pos = targets_orig[gnss_idx, :2] + np.random.normal(0, config.gnss_noise_std, (len(gnss_idx),2))
    gnss_valid = np.ones(len(gnss_idx), dtype=bool)

    bevs = [np.zeros((1,1)) for _ in range(num_frames)]
    return bevs, imu_data, gnss_pos, cam_imgs, targets_norm, lidar_pts, pos_mean, pos_std