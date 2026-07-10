import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from src.data.kitti_dataset import KITTIRawDataset


def main():
    print("=" * 80)
    print("CHECK TRAINING DATA AND TARGET MOTION")
    print("=" * 80)

    train_dataset = KITTIRawDataset(
        Config.train_sequence_dirs,
        max_frames=Config.max_frames
    )

    test_dataset = KITTIRawDataset(
        [Config.test_sequence_dir],
        max_frames=Config.max_frames
    )

    print("Training samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    targets = []

    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        targets.append(sample["target"].numpy())

    targets = np.asarray(targets)

    names = ["dx_body", "dy_body", "dv", "dyaw"]

    print("\nTarget motion statistics from training data")
    print("-" * 80)

    for i, name in enumerate(names):
        values = targets[:, i]

        print(name)
        print("  min :", float(np.min(values)))
        print("  max :", float(np.max(values)))
        print("  mean:", float(np.mean(values)))
        print("  std :", float(np.std(values)))
        print()

    print("First 10 target motions:")
    print("[dx_body, dy_body, dv, dyaw]")
    print(targets[:10])

    print("\nMeaning:")
    print("dx_body = forward movement between two frames")
    print("dy_body = sideways movement")
    print("dv      = speed change")
    print("dyaw    = heading change")
    print("\nIf these values are very small or unstable, training must use normalization.")

    print("\nDONE")


if __name__ == "__main__":
    main()