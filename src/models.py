import torch
import torch.nn as nn
import torchvision.models as models

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding='same', dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.1)
    def forward(self, x):
        return self.drop(self.relu(self.bn(self.conv(x))))

class IMUEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, kernel_size=5, num_layers=4):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding='same')
        self.blocks = nn.ModuleList([TCNBlock(hidden_dim, hidden_dim, kernel_size, 2**i) for i in range(num_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv1(x)
        for blk in self.blocks:
            x = blk(x)
        return self.pool(x).squeeze(-1)

class PointNet(nn.Module):
    def __init__(self, point_dim=3, feat_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(point_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, feat_dim), nn.ReLU()
        )
    def forward(self, x):
        x = self.mlp(x)
        return torch.max(x, dim=1)[0]

class CameraEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, out_dim)
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    def forward(self, imu, lidar, cam):
        stacked = torch.stack([imu, lidar, cam], dim=1)
        out, _ = self.attn(stacked, stacked, stacked)
        out = self.norm(out + stacked)
        return out.mean(dim=1)

class FusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.imu_enc = IMUEncoder(config.imu_input_dim, config.imu_hidden_dim,
                                   config.imu_tcn_kernel, config.imu_tcn_layers)
        self.lidar_enc = PointNet(config.lidar_point_dim, config.lidar_feat_dim)
        self.cam_enc = CameraEncoder(config.camera_feat_dim)
        self.proj_imu = nn.Linear(config.imu_hidden_dim, config.common_dim)
        self.proj_lidar = nn.Linear(config.lidar_feat_dim, config.common_dim)
        self.proj_cam = nn.Linear(config.camera_feat_dim, config.common_dim)
        self.attn = CrossAttention(config.common_dim, config.num_attention_heads)
        self.head = nn.Sequential(
            nn.Linear(config.common_dim, 128), nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.output_dim)
        )
    def forward(self, imu, lidar, cam):
        imu_f = self.proj_imu(self.imu_enc(imu))
        lidar_f = self.proj_lidar(self.lidar_enc(lidar))
        cam_f = self.proj_cam(self.cam_enc(cam))
        fused = self.attn(imu_f, lidar_f, cam_f)
        return self.head(fused)   # only state, no bias