from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
MANIFEST_PATH = RESULTS_ROOT / "new_frame_pair_manifest.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "canonical_epoch3" / "best_model.pth"

CHECKPOINT_SHA256 = (
    "ff210612469f6731dde7e5740aa84e91419a1470acc7a30585132a8aa4fd5f06"
)

SPLIT_COUNTS = {
    "train": 19483,
    "validation": 2121,
    "diagnostic_0018": 269,
    "final_test": 2197,
}


class Config:
    project_root = PROJECT_ROOT
    data_root = DATA_ROOT
    results_root = RESULTS_ROOT
    manifest_path = MANIFEST_PATH
    checkpoint_path = CHECKPOINT_PATH
    checkpoint_sha256 = CHECKPOINT_SHA256
    split_counts = SPLIT_COUNTS

    camera_size = (160, 96)
    lidar_bev_size = 128
    imu_window_shape = (5, 6)

    model_inputs = (
        "camera_t",
        "camera_t1",
        "lidar_t",
        "lidar_t1",
        "imu_window",
    )
    model_outputs = ("dx_m", "dy_m")
    model_output_dim = 2
    model_parameter_count = 785_346
    model_outputs_normalized = False
    selected_epoch = 3
    seed = 42
