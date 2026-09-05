# Thesis Results Tables

All reported results correspond to the frozen canonical SensorFusionMotionModel checkpoint selected using validation recursive ATE RMSE.

## Table 1. Canonical Model Configuration

| Item | Value |
|---|---|
| Canonical model | SensorFusionMotionModel |
| Model source | `src/models/sensor_fusion.py` |
| Model parameters | 785,346 |
| Neural output | Local relative translation `[dx, dy]` |
| Camera input | `camera_t`, `camera_t1` |
| LiDAR input | `lidar_t`, `lidar_t1` |
| IMU input | 5 x 6 IMU window |
| Heading propagation | Raw IMU `wz` trapezoidal integration |
| Initial global state | Latitude, longitude and yaw once per contiguous sequence |
| Selected epoch | 3 |
| Validation selection metric | Recursive ATE RMSE |
| Frozen checkpoint SHA256 | `ff210612469f6731dde7e5740aa84e91419a1470acc7a30585132a8aa4fd5f06` |

## Table 2. Dataset Split

| Split | Drives | Valid frame pairs | Role |
|---|---:|---:|---|
| Training | 16 | 19,483 | Gradient-based training |
| Validation | 3 | 2,121 | Checkpoint selection and diagnostics |
| Diagnostic Drive 0018 | 1 | 269 | Historical diagnostic only |
| Final test | 3 | 2,197 | Final frozen-model evaluation |
| **Total** | **23** | **24,070** | — |

## Table 3. Final-Test Performance

| Drive | Pairs | Local RMSE (m) | GT distance (m) | Pred. distance (m) | Pred/GT | Recursive ATE (m) | Final error (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0019 | 480 | 0.197 | 405.66 | 445.90 | 1.099 | 26.560 | 41.287 |
| 0046 | 124 | 0.077 | 47.51 | 44.72 | 0.941 | 2.256 | 2.561 |
| 0033 | 1593 | 0.091 | 1705.44 | 1731.35 | 1.015 | 10.312 | 8.918 |
| **Aggregate** | **2197** | **0.122** | **2158.61** | **2221.98** | **1.029** | **15.216** | — |

## Table 4. Validation Drift Decomposition

| Drive | GT distance (m) | Full system ATE (m) | GT translation + IMU yaw ATE (m) | Predicted translation + GT yaw ATE (m) | IMU yaw MAE (deg) |
|---|---:|---:|---:|---:|---:|
| 0022 | 514.80 | 6.232 | 1.076 | 6.296 | 0.483 |
| 0079 | 26.19 | 3.338 | 0.037 | 3.335 | 0.231 |
| 0034 | 919.79 | 7.526 | 2.234 | 8.162 | 0.400 |

## Table 5. Drift-Reduction Experiments

| Experiment | Validation recursive ATE (m) | Decision |
|---|---:|---|
| Frozen canonical checkpoint | **6.914** | **Retained** |
| Sequence fine-tuning with frozen BatchNorm | 9.923 | Rejected |
| Longitudinal-bias full-network probe | 16.030 | Rejected |
| Multi-window gradient aggregation | 9.713 | Rejected |
| Output-layer-only calibration | 10.596 | Rejected |
| Temporal residual GRU | 9.239 | Rejected |
| Stateful temporal residual GRU with TBPTT | 10.352 | Rejected |

## Table 6. Key Aggregate Metrics

| Metric | Validation | Final test |
|---|---:|---:|
| Pairs | 2121 | 2197 |
| Local translation RMSE (m) | 0.094 | 0.122 |
| Step RMSE (m) | 0.094 | 0.121 |
| Recursive ATE RMSE (m) | 6.914 | 15.216 |
| Recursive ATE mean (m) | 6.243 | 12.049 |
| Recursive ATE median (m) | 6.556 | 10.649 |
| Recursive ATE maximum (m) | 13.393 | 41.291 |

## Thesis Interpretation Notes

- Translation error, particularly systematic longitudinal `dx` bias, was the dominant source of recursive drift.
- IMU yaw integration contributed substantially less error than learned translation on the validation drives.
- The direction of longitudinal bias varied across drives, preventing a reliable single global scale correction.
- Multiple drift-reduction approaches improved or preserved local RMSE but degraded recursive cross-drive ATE.
- Therefore the originally selected frozen checkpoint was retained as the canonical final model.

- The final-test aggregate travelled-distance ratio was 1.029, showing good overall motion-scale calibration despite drive-dependent accumulated trajectory drift.
