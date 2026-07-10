import torch
import torch.nn as nn

from src.models.camera_cnn import CameraCNN
from src.models.lidar_cnn import LidarCNN
from src.models.imu_encoder import IMUEncoder


class FusionPINN(nn.Module):
    """
    Inputs:
        camera    : [B, 3, 128, 416]
        lidar_bev : [B, 3, 256, 256]
        imu       : [B, 6]

    Encoded features:
        f_cam : [B, 64]
        f_lid : [B, 64]
        f_imu : [B, 128]

    Fused:
        z : [B, 256]

    Output:
        predicted_motion : [B, 4]
        [dx_body, dy_body, dv, dyaw]
    """

    def __init__(self, config):
        super().__init__()

        self.camera_cnn = CameraCNN(output_dim=config.camera_feat_dim)
        self.lidar_cnn = LidarCNN(output_dim=config.lidar_feat_dim)

        self.imu_encoder = IMUEncoder(
            input_dim=config.imu_input_dim,
            output_dim=config.imu_hidden_dim,
        )

        fused_dim = (
            config.camera_feat_dim
            + config.lidar_feat_dim
            + config.imu_hidden_dim
        )

        self.fusion_net = nn.Sequential(
            nn.Linear(fused_dim, config.common_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),

            nn.Linear(config.common_dim, config.common_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),

            nn.Linear(config.common_dim, config.output_dim),
        )

    def forward(self, camera, lidar_bev, imu):
        f_cam = self.camera_cnn(camera)
        f_lid = self.lidar_cnn(lidar_bev)
        f_imu = self.imu_encoder(imu)

        z = torch.cat([f_cam, f_lid, f_imu], dim=1)

        return self.fusion_net(z)

    def forward_with_features(self, camera, lidar_bev, imu):
        f_cam = self.camera_cnn(camera)
        f_lid = self.lidar_cnn(lidar_bev)
        f_imu = self.imu_encoder(imu)

        z = torch.cat([f_cam, f_lid, f_imu], dim=1)

        predicted_motion = self.fusion_net(z)

        return {
            "f_cam": f_cam,
            "f_lid": f_lid,
            "f_imu": f_imu,
            "z": z,
            "predicted_motion": predicted_motion,
        }
