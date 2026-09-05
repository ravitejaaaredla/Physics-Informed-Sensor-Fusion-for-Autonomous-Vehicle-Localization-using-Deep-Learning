from pathlib import Path
import sys

import torch
import torch.nn.functional as F


ROOT = Path.cwd()

sys.path.insert(
    0,
    str(ROOT),
)


from src.data.kitti_frame_pairs import (
    KittiFramePairDataset,
)

from src.models.sensor_fusion import (
    SensorFusionMotionModel,
)


print()
print("=" * 80)
print("SENSOR-FUSION MODEL SMOKE TEST")
print("=" * 80)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print(
    "Device:",
    device,
)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


dataset = KittiFramePairDataset(
    split="train",
    root=ROOT,
    camera_size=(160, 96),
    bev_size=128,
    imu_window=5,
    cache_lidar=True,
)


sample = dataset[0]


def batch(x):
    return x.unsqueeze(0).to(
        device
    )


camera_t = batch(
    sample["camera_t"]
)

camera_t1 = batch(
    sample["camera_t1"]
)

lidar_t = batch(
    sample["lidar_t"]
)

lidar_t1 = batch(
    sample["lidar_t1"]
)

imu_window = batch(
    sample["imu_window"]
)

target = batch(
    sample["target_dxdy"]
)


model = SensorFusionMotionModel().to(
    device
)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

trainable_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print()
print(
    "Total parameters    :",
    f"{parameter_count:,}",
)

print(
    "Trainable parameters:",
    f"{trainable_count:,}",
)


model.train()

prediction = model(
    camera_t,
    camera_t1,
    lidar_t,
    lidar_t1,
    imu_window,
)


assert prediction.shape == (
    1,
    2,
)

assert torch.isfinite(
    prediction
).all()


print()
print(
    "Initial prediction:",
    [
        round(
            float(x),
            6,
        )
        for x in prediction[0]
    ],
)

print(
    "Target dx/dy      :",
    [
        round(
            float(x),
            6,
        )
        for x in target[0]
    ],
)


loss = F.smooth_l1_loss(
    prediction,
    target,
)


assert torch.isfinite(
    loss
)


print(
    "Initial loss      :",
    f"{float(loss):.6f}",
)


model.zero_grad(
    set_to_none=True
)

loss.backward()


groups = {
    "camera": [],
    "lidar": [],
    "imu": [],
    "fusion": [],
}


for name, parameter in (
    model.named_parameters()
):

    if parameter.grad is None:
        continue

    grad_norm = float(
        parameter.grad.norm()
    )

    if name.startswith(
        "camera_encoder"
    ):
        groups[
            "camera"
        ].append(
            grad_norm
        )

    elif name.startswith(
        "lidar_encoder"
    ):
        groups[
            "lidar"
        ].append(
            grad_norm
        )

    elif name.startswith(
        "imu_encoder"
    ):
        groups[
            "imu"
        ].append(
            grad_norm
        )

    elif name.startswith(
        "fusion"
    ):
        groups[
            "fusion"
        ].append(
            grad_norm
        )


print()
print("=" * 80)
print("GRADIENT CHECK")
print("=" * 80)


for group, values in groups.items():

    assert values, (
        f"No gradients for {group}"
    )

    assert all(
        torch.isfinite(
            torch.tensor(values)
        )
    )

    total = sum(values)

    print(
        f"{group:<10}: "
        f"parameters_with_grad="
        f"{len(values):3d} "
        f"sum_grad_norm="
        f"{total:.8f}"
    )

    assert total > 0.0, (
        f"Zero gradient in {group}"
    )


# ------------------------------------------------------------
# Critical no-GPS / no-position-input assertion.
# ------------------------------------------------------------

forward_names = (
    model.forward.__code__
    .co_varnames[
        :model.forward.__code__.co_argcount
    ]
)

for forbidden in (
    "lat",
    "lon",
    "latitude",
    "longitude",
):
    assert forbidden not in (
        x.lower()
        for x in forward_names
    )


print()
print("=" * 80)
print("MODEL INPUTS")
print("=" * 80)

for name in forward_names:

    if name != "self":
        print(
            " ",
            name,
        )


print()
print(
    "PASS: no latitude/longitude "
    "accepted by model.forward()"
)

print(
    "PASS: forward output is finite"
)

print(
    "PASS: gradients reach camera, "
    "LiDAR, IMU and fusion branches"
)

print()
print("=" * 80)
print(
    "PASS: SENSOR-FUSION MOTION MODEL VERIFIED"
)
print("=" * 80)
