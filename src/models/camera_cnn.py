import torch.nn as nn


class CameraCNN(nn.Module):
    """
    Camera image CNN.

    Input:
        [B, 3, 128, 416]

    Layers:
        Conv stride 2: [B, 16, 64, 208]
        Conv stride 2: [B, 32, 32, 104]
        Conv stride 2: [B, 64, 16, 52]
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

    def forward(self, camera):
        x = self.cnn(camera)
        x = x.flatten(start_dim=1)
        return self.fc(x)
