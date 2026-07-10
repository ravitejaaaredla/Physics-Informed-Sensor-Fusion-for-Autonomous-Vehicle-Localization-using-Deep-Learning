import torch
import torch.nn.functional as F


def compute_physics_loss(predicted_motion):
    """
    predicted_motion:
        [B, 4] = [dx_body, dy_body, dv, dyaw]

    This loss keeps motion physically reasonable.
    """

    dx_body = predicted_motion[:, 0]
    dy_body = predicted_motion[:, 1]
    dv = predicted_motion[:, 2]
    dyaw = predicted_motion[:, 3]

    negative_forward_loss = torch.mean(F.relu(-dx_body) ** 2)
    lateral_loss = torch.mean(dy_body ** 2)
    velocity_smooth_loss = torch.mean(dv ** 2)
    yaw_smooth_loss = torch.mean(dyaw ** 2)

    physics_loss = (
        negative_forward_loss
        + lateral_loss
        + velocity_smooth_loss
        + yaw_smooth_loss
    )

    return physics_loss


def normalize_target(values, target_mean, target_std):
    """
    Normalize motion values so all 4 outputs are learned fairly.

    Without this:
        dx_body dominates the loss.

    With this:
        dx_body, dy_body, dv, dyaw are all important.
    """

    mean = torch.tensor(
        target_mean,
        dtype=values.dtype,
        device=values.device
    )

    std = torch.tensor(
        target_std,
        dtype=values.dtype,
        device=values.device
    )

    return (values - mean) / std


def compute_total_loss(
    predicted_motion,
    target_motion,
    target_mean,
    target_std,
    lambda_data=1.0,
    lambda_physics=0.2,
):
    """
    Model output is still real physical motion:

        [dx_body, dy_body, dv, dyaw]

    But data loss is calculated after normalization.
    This improves training.
    """

    predicted_normalized = normalize_target(
        predicted_motion,
        target_mean,
        target_std
    )

    target_normalized = normalize_target(
        target_motion,
        target_mean,
        target_std
    )

    data_loss = F.mse_loss(
        predicted_normalized,
        target_normalized
    )

    physics_loss = compute_physics_loss(
        predicted_motion
    )

    total_loss = (
        lambda_data * data_loss
        + lambda_physics * physics_loss
    )

    return {
        "total_loss": total_loss,
        "data_loss": data_loss,
        "physics_loss": physics_loss,
    }