"""Small, NPU-friendly CNN: only Conv/BN/ReLU/MaxPool/GAP/Linear (all supported by
Hailo's compiler and ONNX INT8). ~60k params at width=16."""
import torch
import torch.nn as nn


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class WaferCNN(nn.Module):
    def __init__(self, num_classes=8, width=16, in_ch=3):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            block(in_ch, w), nn.MaxPool2d(2),
            block(w, 2 * w), nn.MaxPool2d(2),
            block(2 * w, 4 * w), nn.MaxPool2d(2),
            block(4 * w, 4 * w))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4 * w, num_classes)  # logits; sigmoid at inference (multi-label)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


class FocalBCE(nn.Module):
    """Focal loss for multi-label imbalance. pos_weight optional (per-class)."""
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma, self.pw = gamma, pos_weight

    def forward(self, logits, y):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=self.pw)
        p = torch.sigmoid(logits)
        pt = p * y + (1 - p) * (1 - y)
        return (((1 - pt) ** self.gamma) * bce).mean()
