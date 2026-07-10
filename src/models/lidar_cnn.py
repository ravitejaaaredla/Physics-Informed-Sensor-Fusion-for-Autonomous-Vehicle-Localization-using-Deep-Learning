import torch.nn as nn


class LidarCNN(nn.Module):
    """
    LiDAR BEV CNN.

    Input:
        [B, 3, 256, 256]

    Layers:
        Conv stride 2: [B, 16, 128, 128]
        Conv stride 2: [B, 32, 64, 64]
        Conv stride 2: [B, 64, 32, 32]
        Global average pooling: [B, 64, 1, 1]
        Flatten: [B, 64]

    Output:
        [B, 64]
    """

    def __init__(self, output_dim=64):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(64, output_dim)

    def forward(self, lidar_bev):
        x = self.cnn(lidar_bev)
        x = x.flatten(start_dim=1)
        return self.fc(x)
