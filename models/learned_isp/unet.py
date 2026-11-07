"""Metadata-conditioned U-Net: packed Bayer (4ch) -> full-resolution camera-rendered sRGB.
STATUS: architecture only; weights and image-quality performance are unvalidated.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GELU(),
    )


class FiLM(nn.Module):
    def __init__(self, meta_dim, channels):
        super().__init__()
        self.fc = nn.Linear(meta_dim, channels * 2)

    def forward(self, x, meta):
        gamma, beta = self.fc(meta).chunk(2, dim=1)
        return x * (1 + gamma[..., None, None]) + beta[..., None, None]


class AuroraUNet(nn.Module):
    def __init__(self, base=24, meta_dim=5, use_meta=True):
        super().__init__()
        channels = [base, base * 2, base * 4]
        self.use_meta = use_meta
        self.e1 = block(4, channels[0])
        self.e2 = block(channels[0], channels[1])
        self.e3 = block(channels[1], channels[2])
        self.film = FiLM(meta_dim, channels[2]) if use_meta else None
        self.d2 = block(channels[2] + channels[1], channels[1])
        self.d1 = block(channels[1] + channels[0], channels[0])
        self.up_rgb = nn.Conv2d(channels[0], 12, 1)

    def forward(self, packed, meta=None):
        if packed.ndim != 4 or packed.shape[1] != 4:
            raise ValueError("Expected packed Bayer tensor [N,4,H/2,W/2]")
        if min(packed.shape[-2:]) < 4:
            raise ValueError("Packed Bayer dimensions must both be >= 4 (mosaic >= 8x8)")
        if not torch.isfinite(packed).all():
            raise ValueError("Packed Bayer tensor contains nonfinite values")
        if self.film is not None:
            if meta is None:
                raise ValueError("Metadata-conditioned model requires [N, meta_dim] metadata")
            if meta.ndim != 2 or meta.shape != (packed.shape[0], self.film.fc.in_features):
                raise ValueError("Invalid metadata tensor shape")
            if not torch.isfinite(meta).all():
                raise ValueError("Metadata contains nonfinite values")
        x1 = self.e1(packed)
        x2 = self.e2(F.max_pool2d(x1, 2))
        x3 = self.e3(F.max_pool2d(x2, 2))
        if self.film is not None:
            x3 = self.film(x3, meta)
        y = self.d2(torch.cat([F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False), x2], 1))
        y = self.d1(torch.cat([F.interpolate(y, size=x1.shape[-2:], mode="bilinear", align_corners=False), x1], 1))
        return torch.sigmoid(F.pixel_shuffle(self.up_rgb(y), 2))


def count_params(model):
    return sum(parameter.numel() for parameter in model.parameters())
