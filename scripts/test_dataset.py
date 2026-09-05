from pathlib import Path
import math
import sys

import torch


ROOT = Path.cwd()

sys.path.insert(
    0,
    str(ROOT),
)

from src.data.kitti_frame_pairs import (
    KittiFramePairDataset,
    MODEL_INPUT_KEYS,
)


print()
print("=" * 80)
print("FRAME-PAIR DATASET SMOKE TEST")
print("=" * 80)

print()
print(
    "PyTorch:",
    torch.__version__,
)

print(
    "CUDA available:",
    torch.cuda.is_available(),
)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


expected_lengths = {
    "train": 19483,
    "validation": 2121,
    "diagnostic_0018": 269,
    "final_test": 2197,
}


for split, expected in expected_lengths.items():

    print()
    print("-" * 80)
    print(
        split.upper()
    )
    print("-" * 80)

    dataset = KittiFramePairDataset(
        split=split,
        root=ROOT,
        camera_size=(160, 96),
        bev_size=128,
        imu_window=5,
        cache_lidar=True,
    )

    assert len(dataset) == expected, (
        split,
        len(dataset),
        expected,
    )

    sample = dataset[0]

    assert sample[
        "camera_t"
    ].shape == (
        3,
        96,
        160,
    )

    assert sample[
        "camera_t1"
    ].shape == (
        3,
        96,
        160,
    )

    assert sample[
        "lidar_t"
    ].shape == (
        3,
        128,
        128,
    )

    assert sample[
        "lidar_t1"
    ].shape == (
        3,
        128,
        128,
    )

    assert sample[
        "imu_window"
    ].shape == (
        5,
        6,
    )

    assert sample[
        "target_dxdy"
    ].shape == (
        2,
    )

    for key in MODEL_INPUT_KEYS:

        tensor = sample[key]

        assert torch.isfinite(
            tensor
        ).all(), key

    target = sample[
        "target_dxdy"
    ]

    target_step = math.hypot(
        float(target[0]),
        float(target[1]),
    )

    gt_step = sample[
        "meta"
    ][
        "gt_step_m"
    ]

    assert abs(
        target_step - gt_step
    ) < 1e-5

    print(
        "samples       :",
        len(dataset),
    )

    print(
        "camera_t      :",
        tuple(
            sample[
                "camera_t"
            ].shape
        ),
    )

    print(
        "camera_t1     :",
        tuple(
            sample[
                "camera_t1"
            ].shape
        ),
    )

    print(
        "lidar_t       :",
        tuple(
            sample[
                "lidar_t"
            ].shape
        ),
    )

    print(
        "lidar_t1      :",
        tuple(
            sample[
                "lidar_t1"
            ].shape
        ),
    )

    print(
        "imu_window    :",
        tuple(
            sample[
                "imu_window"
            ].shape
        ),
    )

    print(
        "target dx/dy  :",
        [
            round(
                float(x),
                6,
            )
            for x in target
        ],
    )

    print(
        "target step   :",
        f"{target_step:.6f} m",
    )

    print(
        "drive/frame   :",
        sample[
            "meta"
        ][
            "drive"
        ],
        (
            f"{sample['meta']['frame_t']}"
            f"->{sample['meta']['frame_t1']}"
        ),
    )


print()
print("=" * 80)
print("MODEL INPUT CONTRACT")
print("=" * 80)

for key in MODEL_INPUT_KEYS:
    print(
        "  ",
        key,
    )

print()
print(
    "Latitude/longitude are NOT model inputs."
)

print(
    "They exist only under sample['meta'] "
    "for supervision/evaluation."
)

print()
print(
    "PASS: SENSOR-ONLY FRAME-PAIR DATASET VERIFIED"
)
