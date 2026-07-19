import torch
import torch.nn as nn
import torch.nn.functional as F


class IMUEncoder(nn.Module):
    """
    Robust 6‑dim IMU encoder with LayerNorm and learnable scaling.
    Accepts 'output_dim' (used by FusionPINN) or 'hidden_dim'.
    """

    def __init__(self, input_dim=6, hidden_dim=256, dropout=0.2, output_dim=None):
        super().__init__()
        # If output_dim is provided, use it as the final dimension
        final_dim = output_dim if output_dim is not None else hidden_dim

        # Expanding layers with LayerNorm
        self.fc1 = nn.Linear(input_dim, 128)
        self.ln1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 256)
        self.ln2 = nn.LayerNorm(256)

        self.fc3 = nn.Linear(256, final_dim)
        self.ln3 = nn.LayerNorm(final_dim)

        # Learnable scaling factor
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        return x * self.scale