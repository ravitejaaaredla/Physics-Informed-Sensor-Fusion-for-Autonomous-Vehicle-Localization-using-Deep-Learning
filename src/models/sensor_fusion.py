import torch
import torch.nn as nn


class TemporalPairEncoder(nn.Module):
    """
    Joint temporal encoder.

    Instead of independently encoding t and t+1 and then subtracting
    global feature vectors, the CNN directly sees:

        frame_t
        frame_t1
        frame_t1 - frame_t

    This preserves local spatial-temporal changes.

    Six explicit temporal statistics are also supplied:
        mean absolute difference per channel
        RMS difference per channel
    """

    def __init__(
        self,
        channels_per_frame=3,
        output_dim=192,
    ):
        super().__init__()

        temporal_channels = (
            channels_per_frame * 3
        )

        self.backbone = nn.Sequential(

            nn.Conv2d(
                temporal_channels,
                32,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                96,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                96,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
        )

        statistics_dim = (
            channels_per_frame * 2
        )

        self.project = nn.Sequential(

            nn.Linear(
                128 + statistics_dim,
                256,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.10),

            nn.Linear(
                256,
                output_dim,
            ),

            nn.ReLU(inplace=True),
        )


    def forward(
        self,
        x_t,
        x_t1,
    ):

        difference = (
            x_t1 - x_t
        )

        temporal_input = torch.cat(
            [
                x_t,
                x_t1,
                difference,
            ],
            dim=1,
        )

        feature = self.backbone(
            temporal_input
        ).flatten(1)

        mean_abs_difference = (
            difference
            .abs()
            .mean(
                dim=(2, 3)
            )
        )

        rms_difference = torch.sqrt(
            difference.pow(2)
            .mean(
                dim=(2, 3)
            )
            + 1e-8
        )

        temporal_statistics = (
            torch.cat(
                [
                    mean_abs_difference,
                    rms_difference,
                ],
                dim=1,
            )
        )

        combined = torch.cat(
            [
                feature,
                temporal_statistics,
            ],
            dim=1,
        )

        return self.project(
            combined
        )


class IMUGRUEncoder(nn.Module):

    def __init__(
        self,
        input_dim=6,
        hidden_dim=128,
        output_dim=128,
    ):
        super().__init__()

        # Input must already be normalized using
        # TRAIN-SPLIT statistics only.

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.project = nn.Sequential(

            nn.Linear(
                hidden_dim,
                output_dim,
            ),

            nn.ReLU(inplace=True),
        )


    def forward(
        self,
        imu_window,
    ):

        _, hidden = self.gru(
            imu_window
        )

        return self.project(
            hidden[-1]
        )


class SensorFusionMotionModel(nn.Module):
    """
    Sensor-only frame-to-frame motion network.

    Inputs
    ------
    camera_t
    camera_t1
    lidar_t
    lidar_t1
    imu_window

    Output
    ------
    [dx, dy] in vehicle coordinates, metres.

    No latitude, longitude, GPS position,
    or absolute heading enter the network.
    """

    def __init__(self):
        super().__init__()

        self.camera_encoder = (
            TemporalPairEncoder(
                channels_per_frame=3,
                output_dim=192,
            )
        )

        self.lidar_encoder = (
            TemporalPairEncoder(
                channels_per_frame=3,
                output_dim=192,
            )
        )

        self.imu_encoder = (
            IMUGRUEncoder(
                input_dim=6,
                hidden_dim=128,
                output_dim=128,
            )
        )

        fusion_dim = (
            192
            + 192
            + 128
        )

        self.fusion = nn.Sequential(

            nn.Linear(
                fusion_dim,
                256,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.15),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                128,
                2,
            ),
        )

        self._initialize_output()


    def _initialize_output(
        self,
    ):

        final = self.fusion[-1]

        nn.init.normal_(
            final.weight,
            mean=0.0,
            std=1e-3,
        )

        nn.init.zeros_(
            final.bias
        )


    def forward(
        self,
        camera_t,
        camera_t1,
        lidar_t,
        lidar_t1,
        imu_window,
    ):

        camera_feature = (
            self.camera_encoder(
                camera_t,
                camera_t1,
            )
        )

        lidar_feature = (
            self.lidar_encoder(
                lidar_t,
                lidar_t1,
            )
        )

        imu_feature = (
            self.imu_encoder(
                imu_window
            )
        )

        fused = torch.cat(
            [
                camera_feature,
                lidar_feature,
                imu_feature,
            ],
            dim=1,
        )

        return self.fusion(
            fused
        )

