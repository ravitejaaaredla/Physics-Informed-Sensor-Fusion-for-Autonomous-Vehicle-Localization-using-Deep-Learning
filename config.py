# config.py

class Config:
    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    # Main/default KITTI raw sequence
    seq_dir = "data/Kitti_raw/2011_09_26-3/2011_09_26_drive_0009_sync"

    # Multi-drive evaluation sequences
    raw_sequence_dirs = [
        "data/Kitti_raw/2011_09_26-3/2011_09_26_drive_0009_sync",
        "data/Kitti_raw/2011_09_26-6/2011_09_26_drive_0011_sync",
        "data/Kitti_raw/2011_09_26-5/2011_09_26_drive_0013_sync",
        "data/Kitti_raw/2011_09_26-7/2011_09_26_drive_0014_sync",
    ]

    max_frames = 500
    dt = 0.1

    # ------------------------------------------------------------
    # Windowed temporal training
    # ------------------------------------------------------------
    window_size = 10
    stride = 1

    # ------------------------------------------------------------
    # Sensor dimensions
    # ------------------------------------------------------------
    imu_input_dim = 6

    # Keep this at 1024 if your model was trained with 1024.
    # If you change this, you must retrain the model.
    lidar_num_points = 1024
    lidar_point_dim = 3
    lidar_feat_dim = 64

    # Keep image size same as training.
    # If you change this, retrain.
    camera_img_h = 128
    camera_img_w = 416
    camera_feat_dim = 64

    # ------------------------------------------------------------
    # Model dimensions
    # ------------------------------------------------------------
    imu_hidden_dim = 128
    imu_tcn_kernel = 5
    imu_tcn_layers = 4

    common_dim = 256
    num_attention_heads = 4

    # Model output:
    # [dx_body, dy_body, dv, dyaw]
    output_dim = 4

    dropout = 0.2

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    batch_size = 16
    num_epochs = 80
    learning_rate = 1e-3
    weight_decay = 1e-5

    # ------------------------------------------------------------
    # Loss weights
    # ------------------------------------------------------------
    lambda_data = 1.0
    lambda_physics = 0.2
    lambda_smooth = 0.01

    # ------------------------------------------------------------
    # EKF / Localization
    # ------------------------------------------------------------
    gnss_rate = 10.0
    gnss_noise_std = 1.0

    process_noise_pos = 0.5
    process_noise_vel = 0.5
    process_noise_yaw = 0.1

    # ------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------
    seed = 42
    model_type = "pinn_delta"