"""
Model backbones for Experiment 002.

Two architectures, both with `num_classes=3` as the default (the C5/C8
3-class contract per ../NOTES.md §2.2 Issue A):

  - Compact1DCNN_3Layer: 3 conv stages (16 / 32 / 64 channels, kernels
    15 / 7 / 5, strides 2 / 2 / 2, BatchNorm + ReLU, adaptive
    average-pool to 1) followed by a 64->num_classes linear head.
    Used for R1_Raw and R1_Env (1 x N time-domain inputs).

  - StandardMLP_2Layer: 1024 -> 64 -> 32 -> num_classes with BatchNorm
    and ReLU between layers and 0.2 dropout after the first hidden
    layer. Used for R2_Raw, R2_Env, R3_Order (1-D feature vectors).

The architecture is intentionally small; the experiment_config.json
pins a `≤ 8.2 kB working RAM` embedded resource target and a
`≤ 71 kOps/inference` budget, and these backbones fit well under both.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Compact1DCNN_3Layer(nn.Module):
    """1D-CNN with 3 conv stages, used for R1_Raw / R1_Env.

    Input shape:  (batch, 1, 2048)
    Output shape: (batch, num_classes)
    """

    def __init__(self, in_length: int = 2048, num_classes: int = 3):
        super().__init__()
        self.in_length = in_length
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = z.view(z.size(0), -1)
        return self.fc(z)


class StandardMLP_2Layer(nn.Module):
    """MLP with 2 hidden layers, used for R2_* / R3_Order.

    Input shape:  (batch, 1024)
    Output shape: (batch, num_classes)
    """

    def __init__(self, in_features: int = 1024, num_classes: int = 3, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(representation: str, num_classes: int = 3) -> nn.Module:
    """Construct the model backbone for a given representation name.

    The mapping (representation -> backbone) is:
      R1_Raw, R1_Env -> Compact1DCNN_3Layer
      R2_Raw, R2_Env, R3_Order -> StandardMLP_2Layer
    """
    if representation in ("R1_Raw", "R1_Env"):
        return Compact1DCNN_3Layer(in_length=2048, num_classes=num_classes)
    if representation in ("R2_Raw", "R2_Env", "R3_Order"):
        return StandardMLP_2Layer(in_features=1024, num_classes=num_classes)
    raise ValueError(f"Unknown representation: {representation}")
