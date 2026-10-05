from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class TextGuidedBoxRegressor(nn.Module):
    """Small offline baseline for multimodal visual grounding.

    The model fuses 7 visual channels (RGB + infrared RGB + depth) with a hashed
    English query embedding. It is intentionally compact so the pipeline can be
    validated on CPU before replacing the encoders with larger VLM backbones.
    """

    def __init__(
        self,
        input_channels: int = 7,
        vocab_size: int = 8192,
        text_dim: int = 128,
        width: int = 48,
        min_box_size: float = 1e-3,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.vocab_size = vocab_size
        self.text_dim = text_dim
        self.width = width
        self.min_box_size = min_box_size

        c1, c2, c3, c4 = width, width * 2, width * 3, width * 4
        self.visual = nn.Sequential(
            ConvBlock(input_channels, c1, stride=2),
            ResidualBlock(c1),
            ConvBlock(c1, c2, stride=2),
            ResidualBlock(c2),
            ConvBlock(c2, c3, stride=2),
            ResidualBlock(c3),
            ConvBlock(c3, c4, stride=2),
            ResidualBlock(c4),
        )
        self.embedding = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        self.text_norm = nn.LayerNorm(text_dim)
        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.SiLU(inplace=True),
            nn.Linear(text_dim, text_dim),
        )
        self.text_to_visual = nn.Linear(text_dim, c4)
        self.attention_bias = nn.Conv2d(c4, 1, kernel_size=1)
        self.head = nn.Sequential(
            nn.Linear(c4 * 2 + text_dim, c4),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(c4, c4 // 2),
            nn.SiLU(inplace=True),
            nn.Linear(c4 // 2, 4),
        )

    def config(self) -> dict[str, int | float]:
        return {
            "input_channels": self.input_channels,
            "vocab_size": self.vocab_size,
            "text_dim": self.text_dim,
            "width": self.width,
            "min_box_size": self.min_box_size,
        }

    def encode_text(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        mask = mask.unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.text_mlp(self.text_norm(pooled))

    def forward(self, images: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.visual(images)
        text_features = self.encode_text(tokens, mask)
        query = self.text_to_visual(text_features).unsqueeze(-1).unsqueeze(-1)

        attention_logits = (features * query).sum(dim=1) / math.sqrt(features.shape[1])
        attention_logits = attention_logits + self.attention_bias(features).squeeze(1)
        attention = torch.softmax(attention_logits.flatten(1), dim=1).view_as(attention_logits)

        attended = (features * attention.unsqueeze(1)).sum(dim=(2, 3))
        global_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        fused = torch.cat([attended, global_features, text_features], dim=1)
        raw = self.head(fused)
        center = torch.sigmoid(raw[:, :2])
        size = torch.sigmoid(raw[:, 2:]) * (1.0 - self.min_box_size) + self.min_box_size
        half = size * 0.5
        x1y1 = (center - half).clamp(0.0, 1.0)
        x2y2 = (center + half).clamp(0.0, 1.0)
        return torch.cat([x1y1, x2y2], dim=1)
