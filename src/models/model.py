import torch
import torch.nn as nn
import os

class GhostModule(nn.Module):
    """Ghost module for lightweight feature extraction."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(GhostModule, self).__init__()
        self.primary_conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size, stride, padding)
        self.cheap_operation = nn.Conv2d(out_channels // 2, out_channels // 2, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return self.relu(out)

class GhostFaceNetV2(nn.Module):
    """Lightweight face recognition model for AT&T Database."""
    def __init__(self, num_classes):
        super(GhostFaceNetV2, self).__init__()
        self.features = nn.Sequential(
            GhostModule(1, 16),  # Input: 1 channel (grayscale)
            nn.MaxPool2d(2, 2),  # 112x112 -> 56x56
            GhostModule(16, 32),
            nn.MaxPool2d(2, 2),  # 56x56 -> 28x28
            GhostModule(32, 64),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 14 * 14, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x