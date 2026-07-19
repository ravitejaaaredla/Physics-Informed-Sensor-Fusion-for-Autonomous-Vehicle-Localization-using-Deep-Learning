import torch
import torch.nn.functional as F

def imu_yaw_loss(predicted_dyaw, imu_omega_z, dt=0.1, scale=1.0):
    """
    Supervise dyaw using the IMU gyroscope, amplified by a scale factor.
    """
    dyaw_imu = imu_omega_z * dt * scale   # Apply scaling here!
    return F.mse_loss(predicted_dyaw, dyaw_imu)

def compute_physics_loss(predicted_motion):
    dx_body = predicted_motion[:, 0]
    dy_body = predicted_motion[:, 1]
    dv = predicted_motion[:, 2]
    dyaw = predicted_motion[:, 3]

    negative_forward_loss = torch.mean(F.relu(-dx_body) ** 2)
    lateral_loss = torch.mean(dy_body ** 2)
    velocity_smooth_loss = torch.mean(dv ** 2)
    yaw_smooth_loss = torch.mean(dyaw ** 2)

    return negative_forward_loss + lateral_loss + velocity_smooth_loss + yaw_smooth_loss

def normalize_target(values, target_mean, target_std):
    mean = torch.tensor(target_mean, dtype=values.dtype, device=values.device)
    std = torch.tensor(target_std, dtype=values.dtype, device=values.device)
    return (values - mean) / std

def compute_total_loss(
    predicted_motion,
    target_motion,
    target_mean,
    target_std,
    imu_omega_z=None,
    lambda_data=1.0,
    lambda_physics=1.0,
    lambda_imu=1.0,
    imu_dyaw_scale=1.0,   # <-- NEW parameter
    dt=0.1,
):
    predicted_norm = normalize_target(predicted_motion, target_mean, target_std)
    target_norm = normalize_target(target_motion, target_mean, target_std)

    data_loss = F.mse_loss(predicted_norm, target_norm)
    physics_loss = compute_physics_loss(predicted_motion)

    total_loss = lambda_data * data_loss + lambda_physics * physics_loss

    if imu_omega_z is not None:
        predicted_dyaw = predicted_motion[:, 3]
        # Pass the scale to the IMU loss
        imu_loss = imu_yaw_loss(predicted_dyaw, imu_omega_z, dt, scale=imu_dyaw_scale)
        total_loss = total_loss + lambda_imu * imu_loss
    else:
        imu_loss = torch.tensor(0.0, device=predicted_motion.device)

    return {
        "total_loss": total_loss,
        "data_loss": data_loss,
        "physics_loss": physics_loss,
        "imu_loss": imu_loss,
    }