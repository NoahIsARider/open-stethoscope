"""Shared model definition used at inference time (mirrors train_qc_models.PCGNet)."""
import torch.nn as nn


class PCGNet(nn.Module):
    def __init__(self, n_pos=4, n_mur=3, embed=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head_pos = nn.Sequential(nn.Linear(embed, 128), nn.ReLU(inplace=True), nn.Linear(128, n_pos))
        self.head_mur = nn.Sequential(nn.Linear(embed, 128), nn.ReLU(inplace=True), nn.Linear(128, n_mur))

    def forward(self, x):
        e = self.encoder(x)
        e = self.avgpool(e).flatten(1)
        return self.head_pos(e), self.head_mur(e)
