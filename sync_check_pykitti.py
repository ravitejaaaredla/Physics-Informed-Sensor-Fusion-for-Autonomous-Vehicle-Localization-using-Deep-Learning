import pykitti
import matplotlib.pyplot as plt
import numpy as np
import os

# Point to the parent folder containing all data
basedir = '/Users/ravitejareddy/Desktop/iu campus/Raviteja_Reddy_102302880_thesis/Physics-Informed Sensor Fusion for Autonomous Vehicle Localization using Deep Learning/data'

# Date and drive ID
date = '2011_09_26'
drive = '0018'

print("=" * 80)
print("CREATING SYMLINK FOR pykitti")
print("=" * 80)

# Create symlink so pykitti can find the drive
target = os.path.join(basedir, '2011_09_26-7', '2011_09_26_drive_0018_sync')
link = os.path.join(basedir, '2011_09_26', '2011_09_26_drive_0018_sync')

# Make sure the date folder exists
date_folder = os.path.join(basedir, '2011_09_26')
if not os.path.exists(date_folder):
    os.makedirs(date_folder)
    print(f"Created folder: {date_folder}")

# Create symlink if it doesn't exist
if not os.path.exists(link):
    os.symlink(target, link, target_is_directory=True)
    print(f"✅ Created symlink: {link} -> {target}")
else:
    print(f"✅ Symlink already exists: {link}")

print("\n" + "=" * 80)
print("LOADING KITTI DATASET WITH pykitti")
print("=" * 80)

# Load the dataset
dataset = pykitti.raw(basedir, date, drive)
print(f"✅ Dataset loaded successfully!")
print(f"   Number of frames: {len(dataset)}")
print(f"   Calibration loaded: {dataset.calib is not None}")

# ============================================
# VISUALIZATION: Frame 200
# ============================================
frame_idx = 200
print(f"\n--- Visualizing frame {frame_idx} ---")

# Get camera image and LiDAR points
img = dataset.get_cam2(frame_idx)
lidar = dataset.get_velo(frame_idx)

# Project LiDAR onto camera image
points_lidar = lidar[:, :3]  # x, y, z
points_cam = dataset.calib.T_cam2_velo.dot(np.vstack((points_lidar.T, np.ones(points_lidar.shape[0]))))
points_cam = points_cam[:3, :]  # (3, N)
points_cam = points_cam[:, points_cam[2, :] > 0]  # z > 0

# Project to image plane
points_img = dataset.calib.K_cam2.dot(points_cam)
points_img = points_img / points_img[2, :]
u, v = points_img[0, :].astype(np.int32), points_img[1, :].astype(np.int32)

# Filter within image bounds
h, w = img.size[1], img.size[0]
valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
u, v = u[valid], v[valid]

print(f"   Projected {len(u)} LiDAR points onto image.")

# Plot and save
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.scatter(u, v, s=1, c='red', alpha=0.5)
plt.title(f"Frame {frame_idx}: Camera + LiDAR Projection (Red dots = LiDAR points)")
plt.axis('off')

# Save the figure
output_dir = os.path.join(basedir, '..', 'runs', 'sync_check')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'frame_{frame_idx}_sync_check_pykitti.png')
plt.savefig(output_path, dpi=150)
plt.close()
print(f"✅ Visualization saved to: {output_path}")
print(f"   Open it with: open {output_path}")