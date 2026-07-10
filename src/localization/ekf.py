import math
import numpy as np


class FourStateEKF:
    """
    EKF state:
        [x, y, velocity, yaw]

    Motion input:
        [dx_body, dy_body, dv, dyaw]
    """

    def __init__(self, config):
        self.config = config

        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)

        self.Q = np.diag(
            [
                config.process_noise_pos,
                config.process_noise_pos,
                config.process_noise_vel,
                config.process_noise_yaw,
            ]
        ).astype(np.float64)

        self.R = np.eye(2, dtype=np.float64) * (config.gnss_noise_std ** 2)

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def initialize(self, x, y, velocity, yaw):
        self.x = np.array([x, y, velocity, yaw], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)

    def predict(self, dx_body, dy_body, dv, dyaw):
        x_old = self.x[0]
        y_old = self.x[1]
        v_old = self.x[2]
        yaw_old = self.x[3]

        dx_world = math.cos(yaw_old) * dx_body - math.sin(yaw_old) * dy_body
        dy_world = math.sin(yaw_old) * dx_body + math.cos(yaw_old) * dy_body

        x_new = x_old + dx_world
        y_new = y_old + dy_world
        v_new = v_old + dv
        yaw_new = self.normalize_angle(yaw_old + dyaw)

        self.x = np.array([x_new, y_new, v_new, yaw_new], dtype=np.float64)

        F = np.eye(4, dtype=np.float64)
        F[0, 3] = -math.sin(yaw_old) * dx_body - math.cos(yaw_old) * dy_body
        F[1, 3] = math.cos(yaw_old) * dx_body - math.sin(yaw_old) * dy_body

        self.P = F @ self.P @ F.T + self.Q

        return self.x.copy()

    def update_gnss(self, measured_x, measured_y):
        z = np.array([measured_x, measured_y], dtype=np.float64)

        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        residual = z - H @ self.x

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ residual

        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

        return self.x.copy()
