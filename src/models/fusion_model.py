import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding='same', dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class IMUEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, kernel_size=5, num_layers=4):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding='same')
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            self.blocks.append(TCNBlock(hidden_dim, hidden_dim, kernel_size, dilation))
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1,2)            # (batch, input_dim, seq_len)
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x).squeeze(-1)   # (batch, hidden_dim)
        return x

class PointNet(nn.Module):
    def __init__(self, point_dim=3, feat_dim=64):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(point_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU()
        )
    def forward(self, x):
        # x: (batch, N, point_dim)
        x = self.mlp1(x)                # (batch, N, feat_dim)
        x = torch.max(x, dim=1)[0]      # max pooling
        return x

class ResNetEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        resnet = models.resnet18(weights=None)
        # Modify first conv to accept 3x128x416
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        # x: (batch, 3, H, W)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, imu_feat, lidar_feat, cam_feat):
        # Stack as (batch, 3, dim)
        stacked = torch.stack([imu_feat, lidar_feat, cam_feat], dim=1)
        out, _ = self.attn(stacked, stacked, stacked)
        out = self.norm(out + stacked)
        fused = out.mean(dim=1)       # (batch, dim)
        return fused

class FusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.imu_enc = IMUEncoder(input_dim=config.imu_input_dim, hidden_dim=config.imu_hidden_dim,
                                   kernel_size=config.imu_tcn_kernel, num_layers=config.imu_tcn_layers)
        self.lidar_enc = PointNet(point_dim=config.lidar_point_dim, feat_dim=config.lidar_feat_dim)
        self.cam_enc = ResNetEncoder(out_dim=config.camera_feat_dim)

        # Project to common dimension
        self.proj_imu = nn.Linear(config.imu_hidden_dim, config.common_dim)
        self.proj_lidar = nn.Linear(config.lidar_feat_dim, config.common_dim)
        self.proj_cam = nn.Linear(config.camera_feat_dim, config.common_dim)

        self.attention = CrossModalAttention(config.common_dim, num_heads=config.num_attention_heads)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(config.common_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.output_dim)
        )

    def forward(self, imu, lidar, camera):
        # imu: (batch, seq_len, 6)
        # lidar: (batch, N, 3)
        # camera: (batch, 3, H, W)
        imu_feat = self.imu_enc(imu)
        lidar_feat = self.lidar_enc(lidar)
        cam_feat = self.cam_enc(camera)

        imu_feat = self.proj_imu(imu_feat)
        lidar_feat = self.proj_lidar(lidar_feat)
        cam_feat = self.proj_cam(cam_feat)

        fused = self.attention(imu_feat, lidar_feat, cam_feat)
        output = self.fusion_mlp(fused)   # (batch, 6) : Δp (3) + Δatt (3)
        return output