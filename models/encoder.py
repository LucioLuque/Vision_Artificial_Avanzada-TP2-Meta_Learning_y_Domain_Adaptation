import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.features = nn.Sequential(
            ConvBlock(in_channels, hidden_dim),   # 32 -> 16
            ConvBlock(hidden_dim, hidden_dim),    # 16 -> 8
            ConvBlock(hidden_dim, hidden_dim),    # 8 -> 4
            ConvBlock(hidden_dim, hidden_dim),    # 4 -> 2
        )

        self.projection = nn.Linear(hidden_dim * 2 * 2, embedding_dim)

    def forward(self, x):
        x = self.features(x)                 # [B, 64, 2, 2]
        x = x.flatten(start_dim=1)           # [B, 256]
        x = self.projection(x)               # [B, embedding_dim]
        return x