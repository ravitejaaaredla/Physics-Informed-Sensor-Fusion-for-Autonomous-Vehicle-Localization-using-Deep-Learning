import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.camera_cnn import CameraCNN
from src.models.lidar_cnn import LidarCNN
from src.models.imu_encoder import IMUEncoder


class FusionPINN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.camera_cnn = CameraCNN(config)
        self.lidar_cnn = LidarCNN(config)
        self.imu_encoder = IMUEncoder(
            input_dim=config.imu_input_dim,
            hidden_dim=config.imu_hidden_dim,
            dropout=config.dropout,
            output_dim=config.imu_hidden_dim
        )

        # Fused MLP predicts only [dx, dy, dv] (3 outputs)
        fused_dim = config.camera_feat_dim + config.lidar_feat_dim + config.imu_hidden_dim
        self.fc1 = nn.Linear(fused_dim, config.common_dim)
        self.fc2 = nn.Linear(config.common_dim, config.common_dim // 2)
        self.fc_out = nn.Linear(config.common_dim // 2, 3)  # only 3 outputs

        # Direct IMU -> dyaw (1 output)
        self.imu_to_dyaw = nn.Linear(config.imu_hidden_dim, 1)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, camera, lidar_bev, imu):
        cam_feat = self.camera_cnn(camera)      # [B, 64]
        lidar_feat = self.lidar_cnn(lidar_bev)  # [B, 64]
        imu_feat = self.imu_encoder(imu)        # [B, 256]

        fused = torch.cat([cam_feat, lidar_feat, imu_feat], dim=1)

        x = self.relu(self.fc1(fused))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        out_xyz = self.fc_out(x)  # [B, 3] -> [dx, dy, dv]

        # dyaw comes exclusively from IMU
        dyaw = self.imu_to_dyaw(imu_feat).squeeze(1)  # [B]

        # Concatenate to form [dx, dy, dv, dyaw]
        return torch.cat([out_xyz, dyaw.unsqueeze(1)], dim=1)