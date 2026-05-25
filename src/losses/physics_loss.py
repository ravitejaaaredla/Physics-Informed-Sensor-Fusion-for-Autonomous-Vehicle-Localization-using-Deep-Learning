import torch
import torch.nn as nn

class PhysicsLoss(nn.Module):
    """
    Physics-informed loss based on 2D strapdown INS equations:
        dp/dt = v
        dv/dt = R(ψ) * (a_body) + g
        dψ/dt = ω_z
    """
    def __init__(self, dt, lambda_data=1.0, lambda_physics=0.5, lambda_smooth=0.01):
        super().__init__()
        self.dt = dt
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_smooth = lambda_smooth
        self.mse = nn.MSELoss()

    def forward(self, pred_states, target_states, imu_acc_body, imu_gyro_z, t, mask=None):
        """
        pred_states: (N, 4)   [x, y, v, ψ]   (ψ = heading)
        target_states: (N, 4)
        imu_acc_body: (N, 2)   [ax, ay] in body frame
        imu_gyro_z: (N,)       ω_z
        t: (N,) time vector (for AD)
        mask: (N,) boolean (e.g., available GNSS)
        """
        if mask is None:
            mask = torch.ones_like(t, dtype=torch.bool)

        # Data loss (only where mask=True)
        data_loss = self.mse(pred_states[mask], target_states[mask])

        # Physics residuals via automatic differentiation (requires t as gradient source)
        # We need to compute derivatives with respect to t.
        # For simplicity, use finite differences (more stable with batch training)
        # but we can use torch.autograd.grad for each point if time is input.
        # Here we implement finite differences because it's faster and works.
        # For true AD, we would need to embed time as input to the network.
        # We'll use 2nd-order central differences.
        N = pred_states.shape[0]
        if N < 3:
            physics_loss = torch.tensor(0.0, device=pred_states.device)
        else:
            # Position derivative -> velocity
            dpdt = (pred_states[2:, 0:2] - pred_states[:-2, 0:2]) / (2 * self.dt)
            v_pred = pred_states[1:-1, 2]   # scalar velocity
            res_v = self.mse(dpdt.norm(dim=1), v_pred)

            # Velocity derivative -> acceleration in navigation frame
            dvdt = (pred_states[2:, 2] - pred_states[:-2, 2]) / (2 * self.dt)
            heading = pred_states[1:-1, 3]
            # Rotate body acceleration to navigation frame
            cosψ = torch.cos(heading)
            sinψ = torch.sin(heading)
            a_body = imu_acc_body[1:-1]   # (N-2,2)
            a_nav_x = a_body[:,0] * cosψ - a_body[:,1] * sinψ
            a_nav_y = a_body[:,0] * sinψ + a_body[:,1] * cosψ
            a_nav = torch.stack([a_nav_x, a_nav_y], dim=1)
            g = torch.tensor([0.0, -9.81], device=pred_states.device).expand_as(a_nav)
            a_pred = a_nav + g
            a_target = dvdt.unsqueeze(1)   # (N-2, 2) acceleration in nav frame
            res_a = self.mse(a_pred, a_target)

            # Heading derivative -> gyro_z
            dψdt = (pred_states[2:, 3] - pred_states[:-2, 3]) / (2 * self.dt)
            ω_z_pred = imu_gyro_z[1:-1]
            res_ψ = self.mse(dψdt, ω_z_pred)

            physics_loss = res_v + res_a + res_ψ

        # Smoothness regularisation (2nd order diff of position)
        if N < 3:
            smooth_loss = torch.tensor(0.0, device=pred_states.device)
        else:
            pos = pred_states[:, 0:2]
            d2p = pos[2:] - 2*pos[1:-1] + pos[:-2]
            smooth_loss = self.mse(d2p, torch.zeros_like(d2p))

        total = self.lambda_data * data_loss + self.lambda_physics * physics_loss + self.lambda_smooth * smooth_loss
        return total, data_loss, physics_loss, smooth_loss