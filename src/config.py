import numpy as np
class Config:
    # ------------------------------
    # KITTI data paths and parameters
    # ------------------------------
    kitti_root = "data/kitti/odometry"          # Change to your KITTI root folder
    sequence = "00"
    max_frames = 500                            # Number of frames to load (for quick test)
    seq_len = 10                                # Not directly used; training uses whole trajectory

    # LiDAR BEV grid parameters
    bev_x_range = (-30.0, 30.0)                 # left/right [m]
    bev_y_range = (0.0, 70.0)                   # forward [m]
    bev_z_range = (-2.0, 3.0)                   # height [m]
    bev_resolution = 0.1                        # grid cell size [m]
    bev_height_threshold = 0.5                  # max height for occupancy

    # Sensor rates
    imu_rate = 100.0                            # Hz
    gnss_rate = 10.0                            # Hz
    lidar_rate = 10.0                           # Hz
    camera_rate = 10.0                          # Hz

    # ------------------------------
    # Model dimensions
    # ------------------------------
    imu_input_dim = 6                           # ax, ay, az, gx, gy, gz
    imu_hidden_dim = 128
    imu_tcn_kernel = 5
    imu_tcn_layers = 4

    lidar_num_points = 1024                     # points sampled per frame
    lidar_point_dim = 3
    lidar_feat_dim = 64                         # PointNet output dimension

    camera_img_h = 128
    camera_img_w = 416
    camera_feat_dim = 64                        # ResNet-18 output

    common_dim = 256                            # dimension after projection before attention
    num_attention_heads = 4

    output_dim = 6                              # Δpos (3) + Δatt (3)

    # ------------------------------
    # Training hyperparameters
    # ------------------------------
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-5
    dropout = 0.2

    # Simulated GNSS dropout during training (fraction of measurements removed)
    gnss_dropout_prob = 0.2
    max_outage_sec = 3.0

    # ------------------------------
    # Physics‑informed loss weights
    # ------------------------------
    lambda_data = 1.0
    lambda_physics = 0.5
    lambda_smooth = 0.01

    # Adaptive weighting via learnable parameters (set True to enable)
    adaptive_loss_weights = True

    # ------------------------------
    # EKF parameters
    # ------------------------------
    # Initial IMU biases (random walk standard deviations)
    accel_bias_std = 3e-3 * 9.81                # [m/s^2]
    gyro_bias_std = 10 * (np.pi/180) / 3600     # [rad/s]
    accel_bias_walk = 1e-4 / np.sqrt(imu_rate)
    gyro_bias_walk = 1e-5 / np.sqrt(imu_rate)

    # Initial process noise covariance (continuous)
    process_noise_acc = 120e-6 * 9.81           # [m/s^2/√Hz]
    process_noise_gyro = 0.007 * (np.pi/180)    # [rad/s/√Hz]

    # Measurement noise
    gnss_noise_std = 1.0                        # [m]
    lidar_odom_pos_noise = 0.05                 # [m]
    lidar_odom_att_noise = 0.005                # [rad]

    # EKF initial covariance (diagonal)
    init_pos_std = 1.0                          # [m]
    init_vel_std = 0.5                          # [m/s]
    init_att_std = 0.1                          # [rad]
    init_bias_acc_std = 0.01                    # [m/s^2]
    init_bias_gyro_std = 0.001                  # [rad/s]

    # ------------------------------
    # Dataset specific (auto‑detected later)
    # ------------------------------
    bev_dim = None          # will be set from data
    imu_mean = None
    imu_std = None
    target_mean = None
    target_std = None