import torch
import torch.nn as nn

class PhysicsLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dt = config.lidar_dt
        self.mse = nn.MSELoss()
        self.adaptive = config.adaptive_loss_weights
        if self.adaptive:
            self.log_lambda_data = nn.Parameter(torch.tensor(0.0))
            self.log_lambda_physics = nn.Parameter(torch.tensor(0.0))
            self.log_lambda_smooth = nn.Parameter(torch.tensor(0.0))
            self.log_lambda_bias = nn.Parameter(torch.tensor(0.0))
            self.log_lambda_curv = nn.Parameter(torch.tensor(0.0))
        else:
            self.lambda_data = config.lambda_data
            self.lambda_physics = config.lambda_physics
            self.lambda_smooth = config.lambda_smooth
            self.lambda_bias = 0.01
            self.lambda_curv = 0.01

    def forward(self, pred_state, target_state, imu_acc_body, imu_gyro_z,
                pred_bias=None, dt=None):
        if pred_state.dim() == 2:
            if self.adaptive:
                return torch.exp(-self.log_lambda_data) * self.mse(pred_state, target_state) + self.log_lambda_data
            else:
                return self.lambda_data * self.mse(pred_state, target_state)

        data_loss = self.mse(pred_state, target_state)
        N = pred_state.shape[0]
        if N < 3:
            if self.adaptive:
                return torch.exp(-self.log_lambda_data) * data_loss + self.log_lambda_data
            else:
                return self.lambda_data * data_loss

        pos = pred_state[:, :, :2]
        vel = pred_state[:, :, 2]
        yaw = pred_state[:, :, 3]

        dpdt = (pos[:,2:] - pos[:,:-2]) / (2*self.dt)
        v_mid = vel[:,1:-1]
        res_v = self.mse(dpdt.norm(dim=-1), v_mid)

        dvdt = (vel[:,2:] - vel[:,:-2]) / (2*self.dt)
        a_body = imu_acc_body[:,1:-1]
        cosy = torch.cos(yaw[:,1:-1]); siny = torch.sin(yaw[:,1:-1])
        a_nav_x = a_body[:,:,0]*cosy - a_body[:,:,1]*siny
        a_nav_y = a_body[:,:,0]*siny + a_body[:,:,1]*cosy
        a_nav = torch.stack([a_nav_x, a_nav_y], dim=-1)
        g = torch.tensor([0.,-9.81], device=pred_state.device).expand_as(a_nav)
        a_pred = a_nav + g
        res_a = self.mse(a_pred, dvdt.unsqueeze(-1))

        dyawdt = (yaw[:,2:] - yaw[:,:-2]) / (2*self.dt)
        res_psi = self.mse(dyawdt, imu_gyro_z[:,1:-1])

        physics_loss = res_v + res_a + res_psi

        smooth = self.mse(pos[:,2:] - 2*pos[:,1:-1] + pos[:,:-2],
                          torch.zeros_like(pos[:,2:]))

        curv_loss = 0.0
        if imu_acc_body.shape[1] > 2:
            curv_loss = self.mse(imu_acc_body[:,1:-1,1], vel[:,1:-1] * imu_gyro_z[:,1:-1])

        bias_loss = 0.0
        if pred_bias is not None and pred_bias.dim() == 3:
            bias_res = torch.mean((pred_bias[:,1:] - pred_bias[:,:-1])**2)
            bias_loss = bias_res

        if self.adaptive:
            total = (torch.exp(-self.log_lambda_data) * data_loss +
                     torch.exp(-self.log_lambda_physics) * physics_loss +
                     torch.exp(-self.log_lambda_smooth) * smooth +
                     torch.exp(-self.log_lambda_bias) * bias_loss +
                     torch.exp(-self.log_lambda_curv) * curv_loss +
                     self.log_lambda_data + self.log_lambda_physics +
                     self.log_lambda_smooth + self.log_lambda_bias + self.log_lambda_curv)
        else:
            total = (self.lambda_data * data_loss +
                     self.lambda_physics * physics_loss +
                     self.lambda_smooth * smooth +
                     self.lambda_bias * bias_loss +
                     self.lambda_curv * curv_loss)
        return total