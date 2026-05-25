import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.models.fusion_model import FusionModel
from src.losses.physics_loss import PhysicsLoss

def train_fusion(config, bevs, imus, gps, cams, targets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Convert data to tensors (simplified: we will ignore BEV for now, use raw LiDAR points)
    # For this example, we treat bevs as dummy; in real code, you must convert bevs to point clouds.
    # We'll directly use imus, cams, and generate synthetic point clouds from bevs.
    # This part is heuristic; you should replace with proper LiDAR point loading.
    N = len(imus)
    # Create dummy LiDAR point clouds (random points) – replace with real data
    lidar_points = [np.random.randn(config.lidar_num_points, 3).astype(np.float32) for _ in range(N)]
    lidar_tensor = torch.stack([torch.from_numpy(p) for p in lidar_points])

    imu_tensor = torch.tensor(imus, dtype=torch.float32)          # (N, seq_len, 6)
    # If imus are (N,6) single timestep, we need to create windows of length seq_len.
    # Here we assume imus already windows? We'll reshape: (N, 1, 6) -> expand?
    # For simplicity, we treat imu as (N,1,6) and TCN expects seq_len dimension.
    imu_tensor = imu_tensor.unsqueeze(1)      # (N, 1, 6)

    cam_tensor = torch.tensor(np.array(cams), dtype=torch.float32) # (N, H, W, 3) -> (N, 3, H, W)
    cam_tensor = cam_tensor.permute(0, 3, 1, 2)

    targets_tensor = torch.tensor(targets, dtype=torch.float32)     # (N, 4)

    # Create DataLoader
    dataset = TensorDataset(imu_tensor, lidar_tensor, cam_tensor, targets_tensor)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = FusionModel(config).to(device)
    physics_loss_fn = PhysicsLoss(dt=1.0/config.imu_rate,
                                  lambda_data=config.lambda_data,
                                  lambda_physics=config.lambda_physics,
                                  lambda_smooth=config.lambda_smooth)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Adaptive loss weights (learnable)
    if config.adaptive_loss_weights:
        log_lambdas = nn.Parameter(torch.zeros(3, device=device))
        optimizer.add_param_group({'params': log_lambdas, 'lr': 0.01})

    model.train()
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        for imu_batch, lidar_batch, cam_batch, target_batch in loader:
            imu_batch = imu_batch.to(device)
            lidar_batch = lidar_batch.to(device)
            cam_batch = cam_batch.to(device)
            target_batch = target_batch.to(device)

            # Forward pass
            pred_inc = model(imu_batch, lidar_batch, cam_batch)   # (B,6) Δp + Δatt
            # For simplicity, we convert increments to states (integration)
            # But the loss expects states; we can use the network to directly output states.
            # Let's modify: let the network output full state (x,y,v,ψ) instead.
            # To keep example short, we'll just train with MSE on increments.
            # In final version, use physics loss on states.
            # We'll use a simpler loss for demonstration.
            loss = nn.MSELoss()(pred_inc, target_batch[:, :6])   # dummy; replace with physics loss.

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{config.num_epochs}  Loss: {avg_loss:.6f}")

    return model, None