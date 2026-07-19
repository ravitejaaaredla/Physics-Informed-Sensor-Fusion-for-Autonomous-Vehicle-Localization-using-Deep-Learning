import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config


def read_calib_files(calib_dir):
    """
    Read the KITTI calibration files from the date folder.
    Returns: P2 (3x4) and Tr_velo_to_cam (4x4)
    """
    # Read calib_cam_to_cam.txt to get P_rect_02
    cam_file = calib_dir / 'calib_cam_to_cam.txt'
    velo_file = calib_dir / 'calib_velo_to_cam.txt'

    # Parse P2 (P_rect_02)
    with open(cam_file, 'r') as f:
        for line in f:
            if line.startswith('P_rect_02:'):
                P2 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
                break

    # Parse Tr_velo_to_cam
    with open(velo_file, 'r') as f:
        lines = f.readlines()
        # The format is:
        # calib_time: ...
        # R: ...
        # T: ...
        # We need to combine R (3x3) and T (3x1) into 3x4, then add row [0,0,0,1]
        for i, line in enumerate(lines):
            if line.startswith('R:'):
                R = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 3)
            elif line.startswith('T:'):
                T = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 1)

        Tr_velo_to_cam = np.hstack((R, T))  # 3x4
        Tr_velo_to_cam = np.vstack((Tr_velo_to_cam, np.array([0, 0, 0, 1])))  # 4x4

    return P2, Tr_velo_to_cam


def load_and_project_lidar(lidar_file, P2, Tr_velo_to_cam, img_shape):
    points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    # Transform to camera frame
    points_h = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
    points_cam = (Tr_velo_to_cam @ points_h.T).T
    points_cam = points_cam[points_cam[:, 2] > 0]  # z > 0

    # Project to image plane
    points_img = (P2 @ points_cam.T).T
    points_img = points_img / points_img[:, 2:3]
    u, v = points_img[:, 0].astype(np.int32), points_img[:, 1].astype(np.int32)

    h, w = img_shape
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u[valid], v[valid]


def main():
    frame_idx = 200
    drive_path = Config.test_sequence_dir

    # Calibration directory (date folder)
    calib_dir = Path('data/2011_09_26')
    if not calib_dir.exists():
        print(f"Calibration directory not found: {calib_dir}")
        return

    P2, Tr_velo_to_cam = read_calib_files(calib_dir)

    # Load image
    img_path = drive_path / "image_02" / "data" / f"{frame_idx:010d}.png"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Image not found: {img_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Load LiDAR
    lidar_path = drive_path / "velodyne_points" / "data" / f"{frame_idx:010d}.bin"
    if not lidar_path.exists():
        print(f"LiDAR file not found: {lidar_path}")
        return

    u, v = load_and_project_lidar(lidar_path, P2, Tr_velo_to_cam, (h, w))

    # Plot
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.scatter(u, v, s=1, c='red', alpha=0.5)
    plt.title(f"Frame {frame_idx}: Camera + LiDAR Projection")
    plt.axis('off')

    output_path = Path("runs") / "sync_check" / f"frame_{frame_idx}_sync_check.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Visualization saved to: {output_path}")
    print("   Open it with: open " + str(output_path))


if __name__ == "__main__":
    main()