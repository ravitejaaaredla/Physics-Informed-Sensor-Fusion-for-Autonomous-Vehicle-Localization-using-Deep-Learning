import numpy as np
from pathlib import Path
from PIL import Image

def load_kitti_sequence(config):
    """
    Load KITTI odometry sequence and return:
        bevs   : list of BEV height maps (each as (H,W) numpy array)
        imus   : list of IMU measurements (6,)
        gps    : list of GNSS positions (x,y,z) (z may be 0)
        cams   : list of camera images (H,W,3) or preprocessed features
        targets: list of ground truth [x, y, v, heading] (4,)
    """
    base = Path(config.kitti_root)
    seq = config.sequence

    # KITTI odometry provides ground truth poses (times.txt, poses.txt)
    pose_file = base / "poses" / f"{seq}.txt"
    times_file = base / "times.txt"
    if not pose_file.exists():
        raise FileNotFoundError(f"KITTI ground truth not found at {pose_file}")

    # Load ground truth poses (N x 12) -> each as 3x4 transformation matrix (from camera to world)
    poses = np.loadtxt(pose_file).reshape(-1, 3, 4)
    timestamps = np.loadtxt(times_file)

    # We will use the first `max_frames` frames
    num_frames = min(config.max_frames, len(poses))
    poses = poses[:num_frames]

    # Extract position and heading from pose
    targets = []
    for i in range(num_frames):
        p = poses[i, :3, 3]                     # x, y, z
        # heading from rotation matrix (yaw = atan2(R[1,0], R[0,0]))
        Rmat = poses[i, :3, :3]
        yaw = np.arctan2(Rmat[1,0], Rmat[0,0])
        # Velocity: approximate via finite differences (using dt from timestamps)
        if i == 0:
            v = 0.0
        else:
            dt = timestamps[i] - timestamps[i-1]
            dp = poses[i, :3, 3] - poses[i-1, :3, 3]
            v = np.linalg.norm(dp) / dt if dt > 0 else 0.0
        targets.append([p[0], p[1], v, yaw])

    # ------------------------------
    # Load LiDAR point clouds (Velodyne)
    # ------------------------------
    velo_dir = base / "velodyne" / seq
    bevs = []
    for i in range(num_frames):
        bin_file = velo_dir / f"{i:06d}.bin"
        if not bin_file.exists():
            # If not found, create dummy BEV
            bev = np.zeros((int((config.bev_y_range[1]-config.bev_y_range[0])/config.bev_resolution),
                            int((config.bev_x_range[1]-config.bev_x_range[0])/config.bev_resolution)))
        else:
            points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)   # x,y,z,reflectance
            # Filter points inside ROI
            mask = (points[:,0] >= config.bev_x_range[0]) & (points[:,0] <= config.bev_x_range[1]) & \
                   (points[:,1] >= config.bev_y_range[0]) & (points[:,1] <= config.bev_y_range[1]) & \
                   (points[:,2] >= config.bev_z_range[0]) & (points[:,2] <= config.bev_z_range[1])
            pts = points[mask]
            # Create BEV height map (max height in each cell)
            x_bins = np.arange(config.bev_x_range[0], config.bev_x_range[1]+config.bev_resolution, config.bev_resolution)
            y_bins = np.arange(config.bev_y_range[0], config.bev_y_range[1]+config.bev_resolution, config.bev_resolution)
            x_idx = np.digitize(pts[:,0], x_bins) - 1
            y_idx = np.digitize(pts[:,1], y_bins) - 1
            bev = np.zeros((len(y_bins)-1, len(x_bins)-1), dtype=np.float32)
            # For each cell, store maximum height (z)
            for ix, iy, z in zip(x_idx, y_idx, pts[:,2]):
                if 0 <= ix < bev.shape[1] and 0 <= iy < bev.shape[0]:
                    bev[iy, ix] = max(bev[iy, ix], z)
        bevs.append(bev)

    # ------------------------------
    # Load IMU data (if available) – KITTI raw has IMU, but odometry does not.
    # We will simulate IMU from ground truth with added noise (for training).
    # In real experiments, replace with actual IMU files.
    # Here we create synthetic IMU from ground truth (strapdown equations).
    # (You can replace this with true KITTI IMU if you have the raw dataset.)
    # ------------------------------
    dt = np.mean(np.diff(timestamps[:num_frames]))
    imus = []
    for i in range(num_frames-1):
        # Finite differences for acceleration and angular velocity
        v1 = np.array(targets[i][:2] + [0.0])   # (x,y,0) for velocity? Actually we have v (scalar)
        v2 = np.array(targets[i+1][:2] + [0.0])
        acc = (v2 - v1) / dt
        # Yaw rate
        yaw1 = targets[i][3]
        yaw2 = targets[i+1][3]
        yaw_rate = (yaw2 - yaw1) / dt
        # Add IMU noise
        acc_noise = np.random.normal(0, config.process_noise_acc * np.sqrt(1/dt), 2)
        gyro_noise = np.random.normal(0, config.process_noise_gyro * np.sqrt(1/dt))
        # Simulated IMU reading: [ax, ay, az, gx, gy, gz] (az = 0 in 2D, gx=gy=0)
        imu_meas = [acc[0] + acc_noise[0], acc[1] + acc_noise[1], 0.0,
                    0.0, 0.0, yaw_rate + gyro_noise]
        imus.append(np.array(imu_meas))
    # Duplicate last frame to keep length same
    imus.append(imus[-1])

    # ------------------------------
    # GNSS positions (simulated with noise from ground truth)
    # ------------------------------
    gps = []
    gnss_step = int(config.imu_rate / config.gnss_rate)
    for i in range(num_frames):
        if i % gnss_step == 0:
            pos_noisy = targets[i][:2] + np.random.normal(0, config.gnss_noise_std, 2)
            gps.append([pos_noisy[0], pos_noisy[1], 0.0])
        else:
            gps.append([np.nan, np.nan, np.nan])   # missing measurement
    gps = np.array(gps)

    # ------------------------------
    # Camera images (load if available, else dummy)
    # ------------------------------
    image_dir = base / "image_2" / seq
    cams = []
    for i in range(num_frames):
        img_file = image_dir / f"{i:06d}.png"
        if img_file.exists():
            img = cv2.imread(str(img_file))
            img = cv2.resize(img, (config.camera_img_w, config.camera_img_h))
            img = img / 255.0
        else:
            img = np.zeros((config.camera_img_h, config.camera_img_w, 3))
        cams.append(img)

    # Convert targets to numpy array
    targets = np.array(targets)   # (N,4): x, y, v, heading

    return bevs, imus, gps, cams, targets