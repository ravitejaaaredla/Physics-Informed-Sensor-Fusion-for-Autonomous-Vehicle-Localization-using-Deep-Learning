# src/raw_kitti_utils.py

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


EARTH_RADIUS = 6378137.0


def wrap_angle_np(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def latlon_to_mercator(lat, lon, scale):
    mx = scale * lon * np.pi * EARTH_RADIUS / 180.0
    my = scale * EARTH_RADIUS * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
    return mx, my


def get_frame_ids(folder: Path, suffix: str):
    files = sorted(folder.glob(f"*.{suffix}"))
    ids = []

    for f in files:
        try:
            ids.append(int(f.stem))
        except ValueError:
            continue

    return set(ids)


def get_valid_frame_ids(root: Path, config):
    velo_dir = root / "velodyne_points" / "data"
    img_dir = root / "image_02" / "data"
    oxts_dir = root / "oxts" / "data"

    if not velo_dir.exists():
        raise FileNotFoundError(f"LiDAR folder not found: {velo_dir}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Camera folder not found: {img_dir}")
    if not oxts_dir.exists():
        raise FileNotFoundError(f"OXTS folder not found: {oxts_dir}")

    lidar_ids = get_frame_ids(velo_dir, "bin")
    camera_ids = get_frame_ids(img_dir, "png")
    oxts_ids = get_frame_ids(oxts_dir, "txt")

    all_ids = sorted(list(lidar_ids & camera_ids & oxts_ids))

    if len(all_ids) == 0:
        raise RuntimeError("No synchronized frame IDs found.")

    blocks = []
    current = [all_ids[0]]

    for a, b in zip(all_ids[:-1], all_ids[1:]):
        if b == a + 1:
            current.append(b)
        else:
            blocks.append(current)
            current = [b]

    blocks.append(current)

    valid_ids = max(blocks, key=len)
    valid_ids = valid_ids[: config.max_frames]

    print("Frame availability:")
    print(f"  lidar:  {len(lidar_ids)}")
    print(f"  camera: {len(camera_ids)}")
    print(f"  oxts:   {len(oxts_ids)}")
    print(f"  common synchronized frames: {len(all_ids)}")
    print(f"  continuous block: {valid_ids[0]} to {valid_ids[-1]}")
    print(f"  using:  {len(valid_ids)} frames")

    if len(valid_ids) < config.window_size + 2:
        raise RuntimeError(
            f"Not enough continuous synchronized frames. "
            f"valid_frames={len(valid_ids)}, window_size={config.window_size}"
        )

    return valid_ids


def load_oxts_packet(oxts_file: Path):
    return np.loadtxt(oxts_file)


def load_raw_sequence(seq_dir, config):
    """
    Loads KITTI raw sequence using OXTS as trajectory source.

    Returns:
        states:    [N, 4] = [x, y, v, yaw]
        lidar_pts: list of [lidar_num_points, 3]
        cam_imgs:  list of [H, W, 3]
        imu_data:  [N, 6] = [ax, ay, az, wx, wy, wz]
    """
    root = Path(seq_dir)

    frame_ids = get_valid_frame_ids(root, config)

    velo_dir = root / "velodyne_points" / "data"
    img_dir = root / "image_02" / "data"
    oxts_dir = root / "oxts" / "data"

    # ------------------------------------------------------------
    # Load OXTS packets
    # KITTI OXTS columns:
    # lat lon alt roll pitch yaw vn ve vf vl vu ax ay az ...
    # ------------------------------------------------------------
    oxts_packets = []

    for frame_id in frame_ids:
        oxts_file = oxts_dir / f"{frame_id:010d}.txt"
        oxts_packets.append(load_oxts_packet(oxts_file))

    oxts_packets = np.asarray(oxts_packets, dtype=np.float64)

    # ------------------------------------------------------------
    # Build local XY trajectory from lat/lon
    # ------------------------------------------------------------
    lat0 = oxts_packets[0, 0]
    lon0 = oxts_packets[0, 1]
    scale = np.cos(lat0 * np.pi / 180.0)

    x0, y0 = latlon_to_mercator(lat0, lon0, scale)

    states = []

    xy_positions = []

    for pkt in oxts_packets:
        lat = pkt[0]
        lon = pkt[1]
        yaw = pkt[5]

        mx, my = latlon_to_mercator(lat, lon, scale)

        x = mx - x0
        y = my - y0

        xy_positions.append([x, y, yaw])

    xy_positions = np.asarray(xy_positions, dtype=np.float32)

    for i in range(len(xy_positions)):
        x = xy_positions[i, 0]
        y = xy_positions[i, 1]
        yaw = xy_positions[i, 2]

        if i == 0:
            v = 0.0
        else:
            frame_gap = frame_ids[i] - frame_ids[i - 1]
            dt = max(1e-6, frame_gap * config.dt)
            dp = xy_positions[i, :2] - xy_positions[i - 1, :2]
            v = float(np.linalg.norm(dp) / dt)

        states.append([x, y, v, yaw])

    states = np.asarray(states, dtype=np.float32)

    # ------------------------------------------------------------
    # LiDAR
    # ------------------------------------------------------------
    lidar_pts = []

    for frame_id in frame_ids:
        bin_file = velo_dir / f"{frame_id:010d}.bin"
        pts = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)[:, :3]

        if len(pts) >= config.lidar_num_points:
            sample_idx = np.random.choice(
                len(pts),
                config.lidar_num_points,
                replace=False,
            )
        else:
            sample_idx = np.random.choice(
                len(pts),
                config.lidar_num_points,
                replace=True,
            )

        lidar_pts.append(pts[sample_idx].astype(np.float32))

    # ------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------
    cam_imgs = []

    for frame_id in frame_ids:
        img_file = img_dir / f"{frame_id:010d}.png"

        img = Image.open(img_file).convert("RGB")
        img = img.resize((config.camera_img_w, config.camera_img_h))
        img = np.asarray(img, dtype=np.float32) / 255.0

        cam_imgs.append(img)

    # ------------------------------------------------------------
    # IMU from OXTS
    # ax, ay, az = columns 11:14
    # wx, wy, wz = columns 17:20
    # ------------------------------------------------------------
    imu_data = []

    for pkt in oxts_packets:
        imu = pkt[11:14].tolist() + pkt[17:20].tolist()
        imu_data.append(imu)

    imu_data = np.asarray(imu_data, dtype=np.float32)

    return states, lidar_pts, cam_imgs, imu_data


def build_delta_targets(states):
    """
    Converts absolute states into delta targets.

    Input:
        states[k] = [x, y, v, yaw]

    Output:
        delta[k] = [dx_body, dy_body, dv, dyaw]
    """
    deltas = []

    for k in range(len(states) - 1):
        x, y, v, yaw = states[k]
        x2, y2, v2, yaw2 = states[k + 1]

        dp_world = np.asarray([x2 - x, y2 - y], dtype=np.float32)

        c = np.cos(yaw)
        s = np.sin(yaw)

        R_w_to_b = np.asarray(
            [
                [c, s],
                [-s, c],
            ],
            dtype=np.float32,
        )

        dp_body = R_w_to_b @ dp_world

        dv = v2 - v
        dyaw = wrap_angle_np(yaw2 - yaw)

        deltas.append([dp_body[0], dp_body[1], dv, dyaw])

    return np.asarray(deltas, dtype=np.float32)


class DeltaWindowDataset(Dataset):
    def __init__(self, states, lidar_pts, cam_imgs, imu_data, config):
        self.states = states
        self.lidar_pts = lidar_pts
        self.cam_imgs = cam_imgs
        self.imu_data = imu_data
        self.config = config

        self.delta_targets = build_delta_targets(states)

        self.ends = list(
            range(
                config.window_size - 1,
                len(states) - 1,
                config.stride,
            )
        )

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, idx):
        end = self.ends[idx]
        start = end - self.config.window_size + 1

        imu_win = self.imu_data[start : end + 1]
        lidar = self.lidar_pts[end]
        cam = self.cam_imgs[end]
        prev_state = self.states[end]
        delta_target = self.delta_targets[end]

        return (
            torch.tensor(imu_win, dtype=torch.float32),
            torch.tensor(lidar, dtype=torch.float32),
            torch.tensor(cam, dtype=torch.float32).permute(2, 0, 1),
            torch.tensor(prev_state, dtype=torch.float32),
            torch.tensor(delta_target, dtype=torch.float32),
        )