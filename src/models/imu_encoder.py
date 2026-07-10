import torch.nn as nn


class IMUEncoder(nn.Module):
    """
    IMU/OXTS encoder.

    Input:
        [B, 6] = [af, al, au, wf, wl, wu]

    Output:
        [B, 128]
    """

    def __init__(self, input_dim=6, output_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, output_dim),
            nn.ReLU(),
        )

    def forward(self, imu):
        return self.encoder(imu)
