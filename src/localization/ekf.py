import numpy as np
from config import Config


class FourStateEKF:
    """
    4-State Extended Kalman Filter for vehicle localization.
    State: [x, y, v, yaw]
    """

    def __init__(self, config: Config):
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 0.1

        self.Q = np.diag(
            [
                config.process_noise_pos,
                config.process_noise_pos,
                config.process_noise_vel,
                config.process_noise_yaw,
            ]
        ).astype(np.float64)

        self.R = np.eye(2, dtype=np.float64) * (config.gnss_noise_std ** 2)

    def initialize(self, x, y, velocity, yaw):
        self.x = np.array([x, y, velocity, yaw], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 0.01

    def predict(self, dx_body, dy_body, dv, dyaw):
        x, y, v, yaw = self.x

        # ========== FIXED ROTATION (Flipped dy_body sign) ==========
        # If the dataset defines dy_body as "Right" instead of "Left",
        # we must use the following rotation.
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # OLD (Wrong for your dataset):
        # dx_global = dx_body * cos_yaw - dy_body * sin_yaw
        # dy_global = dx_body * sin_yaw + dy_body * cos_yaw
        
        # NEW (Corrected sign):
        dx_global = dx_body * cos_yaw + dy_body * sin_yaw
        dy_global = dx_body * sin_yaw - dy_body * cos_yaw
        # ===========================================================

        x_new = x + dx_global
        y_new = y + dy_global
        v_new = v + dv
        yaw_new = yaw + dyaw

        self.x = np.array([x_new, y_new, v_new, yaw_new], dtype=np.float64)

        F = np.eye(4, dtype=np.float64)
        # Jacobian update with the NEW signs
        F[0, 3] = -dx_body * sin_yaw + dy_body * cos_yaw
        F[1, 3] = dx_body * cos_yaw + dy_body * sin_yaw

        self.P = F @ self.P @ F.T + self.Q

        return self.x

    def update_gnss(self, gps_x, gps_y):
        z = np.array([gps_x, gps_y], dtype=np.float64)

        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        z_pred = H @ self.x
        y_innov = z - z_pred
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_innov

        I = np.eye(4)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T

        return self.x
