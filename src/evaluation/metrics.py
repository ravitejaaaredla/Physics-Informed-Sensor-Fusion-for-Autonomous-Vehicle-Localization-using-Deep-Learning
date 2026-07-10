import numpy as np


def compute_ate(predicted_xy, ground_truth_xy):
    predicted_xy = np.asarray(predicted_xy, dtype=np.float64)
    ground_truth_xy = np.asarray(ground_truth_xy, dtype=np.float64)

    return np.linalg.norm(predicted_xy - ground_truth_xy, axis=1)


def compute_rpe(predicted_xy, ground_truth_xy):
    predicted_xy = np.asarray(predicted_xy, dtype=np.float64)
    ground_truth_xy = np.asarray(ground_truth_xy, dtype=np.float64)

    predicted_delta = predicted_xy[1:] - predicted_xy[:-1]
    ground_truth_delta = ground_truth_xy[1:] - ground_truth_xy[:-1]

    return np.linalg.norm(predicted_delta - ground_truth_delta, axis=1)
