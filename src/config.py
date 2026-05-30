import numpy as np

class Config:
    # ------------------------------
    # KITTI data paths and parameters
    # ------------------------------
    kitti_root = "data/kitti/odometry"          # for odometry dataset (unified)
    kitti_raw_root = "data/kitti_raw"           # for raw dataset (with oxts, velodyne, image_02)
    sequence = "00"
    max_frames = 500
    seq_len = 10

    # LiDAR BEV grid parameters (for BEV representation – optional)
    bev_x_range = (-30.0, 30.0)
    bev_y_range = (0.0, 70.0)
    bev_z_range = (-2.0, 3.0)
    bev_resolution = 0.1
    bev_height_threshold = 0.5

    # Sensor rates
    imu_rate = 100.0
    gnss_rate = 10.0
    lidar_rate = 10.0
    camera_rate = 10.0
    lidar_dt = 1.0 / imu_rate

    # ------------------------------
    # Model dimensions
    # ------------------------------
    imu_input_dim = 6
    imu_hidden_dim = 128
    imu_tcn_kernel = 5
    imu_tcn_layers = 4

    lidar_num_points = 1024
    lidar_point_dim = 3
    lidar_feat_dim = 64

    camera_img_h = 128
    camera_img_w = 416
    camera_feat_dim = 64

    common_dim = 256
    num_attention_heads = 4
    output_dim = 4          # [x, y, v, ψ]  – absolute states, not increments

    # ------------------------------
    # Training hyperparameters
    # ------------------------------
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-5
    dropout = 0.2

    gnss_dropout_prob = 0.2
    max_outage_sec = 3.0

    # ------------------------------
    # Physics‑informed loss weights
    # ------------------------------
    lambda_data = 1.0
    lambda_physics = 0.5
    lambda_smooth = 0.01
    adaptive_loss_weights = True

    # ------------------------------
    # EKF parameters (optional, for baseline)
    # ------------------------------
    accel_bias_std = 3e-3 * 9.81
    gyro_bias_std = 10 * (np.pi/180) / 3600
    accel_bias_walk = 1e-4 / np.sqrt(imu_rate)
    gyro_bias_walk = 1e-5 / np.sqrt(imu_rate)

    process_noise_acc = 120e-6 * 9.81
    process_noise_gyro = 0.007 * (np.pi/180)

    gnss_noise_std = 1.0
    lidar_odom_pos_noise = 0.05
    lidar_odom_att_noise = 0.005

    init_pos_std = 1.0
    init_vel_std = 0.5
    init_att_std = 0.1
    init_bias_acc_std = 0.01
    init_bias_gyro_std = 0.001

    # ------------------------------
    # Dataset specific (auto‑detected)
    # ------------------------------
    bev_dim = None
    imu_mean = None
    imu_std = None
    target_mean = None
    target_std = None