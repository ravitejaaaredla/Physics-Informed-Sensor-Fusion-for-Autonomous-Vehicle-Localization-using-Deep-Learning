from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"


def find_drive_dir(drive_id):
    pattern = f"2011_09_26_drive_{drive_id}_sync"
    matches = list(DATA_ROOT.rglob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find KITTI drive folder {pattern} inside {DATA_ROOT}"
        )

    return matches[0]


class Config:
    project_root = PROJECT_ROOT
    data_root = DATA_ROOT

    # KITTI drives
    train_drive_ids = ["0009", "0011", "0013", "0014"]
    val_drive_ids = ["0017"]
    test_drive_id = "0017"

    train_sequence_dirs = [
        find_drive_dir(drive_id)
        for drive_id in train_drive_ids
    ]

    val_sequence_dirs = [
        find_drive_dir(drive_id)
        for drive_id in val_drive_ids
    ]

    test_sequence_dir = find_drive_dir(test_drive_id)

    max_frames = None

    # Camera input
    camera_img_h = 128
    camera_img_w = 416
    camera_feat_dim = 64

    # LiDAR BEV input
    bev_x_range = (0.0, 50.0)
    bev_y_range = (-25.0, 25.0)
    bev_z_range = (-2.5, 2.5)
    bev_shape = (256, 256)
    lidar_feat_dim = 64

    # IMU/OXTS input
    imu_input_dim = 6
    imu_hidden_dim = 128

    # Fusion model
    common_dim = 256
    dropout = 0.2
    output_dim = 4

    # Training
    seed = 42
    batch_size = 16
    learning_rate = 1e-3
    weight_decay = 1e-5

    lambda_data = 1.0
    lambda_physics = 0.2

    # Target normalization
    # Target order: [dx_body, dy_body, dv, dyaw]
    target_mean = [
        0.9036931395530701,
        -0.0014772213762626052,
        -0.012304166331887245,
        -0.00014998042024672031,
    ]

    target_std = [
        0.48915764689445496,
        0.011165588162839413,
        0.07812748104333878,
        0.010136288590729237,
    ]

    # EKF parameters
    process_noise_pos = 0.5
    process_noise_vel = 0.5
    process_noise_yaw = 0.1
    gnss_noise_std = 0.01

    # Output folders
    runs_dir = PROJECT_ROOT / "runs"
    checkpoint_dir = runs_dir / "checkpoints"
    eval_dir = runs_dir / "eval"