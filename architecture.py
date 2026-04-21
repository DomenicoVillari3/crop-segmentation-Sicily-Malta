import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x):
        x = self.up(x)
        return F.relu(self.conv(x) + self.shortcut(x))

class PrithviSegmentation4090(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        d = backbone.out_channels[-1]
        self.decoder = nn.Sequential(
            ResidualUpBlock(d, 256), ResidualUpBlock(256, 128),
            ResidualUpBlock(128, 64), ResidualUpBlock(64, 32),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = self.backbone(x.permute(0, 2, 1, 3, 4))
        tokens = feats[-1][:, 1:, :]
        grid = int(np.sqrt(tokens.shape[1]))
        logits = self.decoder(tokens.transpose(1, 2).reshape(B, -1, grid, grid))
        return F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
