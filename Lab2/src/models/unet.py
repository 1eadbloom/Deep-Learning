"""
UNet（二元語意分割）

論文參考：https://arxiv.org/abs/1505.04597v1

結構概要：
    編碼端四段：每段雙層 conv 再接 max pooling。
    最底層 bottleneck。
    解碼端四段：轉置卷積上採樣、與 skip 串接、再雙層 conv。
    最後 1×1 conv 輸出一個通道的 logits（sigmoid 放在 loss／評估時做）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 基本卷積區塊 ---


class DoubleConv(nn.Module):
    """兩組 Conv-BN-ReLU 串在一起。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# --- 編碼端（下採樣 + 保留 skip）---


class EncoderBlock(nn.Module):
    """編碼：DoubleConv 後接 2×2 max pool。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        pooled = self.pool(skip)
        return pooled, skip


# --- 解碼端（上採樣 + 融合 skip）---


class DecoderBlock(nn.Module):
    """解碼：轉置卷積放大、concat skip、再接 DoubleConv。"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# --- 完整 UNet ---


class UNet(nn.Module):
    """
    典型的四階編碼／四階解碼 UNet。
    通道大致為：64 → 128 → 256 → 512，瓶頸 1024。
    """

    def __init__(self, in_channels=3, out_channels=1, base_features=64):
        super().__init__()
        f = base_features

        self.enc1 = EncoderBlock(in_channels, f)
        self.enc2 = EncoderBlock(f, f * 2)
        self.enc3 = EncoderBlock(f * 2, f * 4)
        self.enc4 = EncoderBlock(f * 4, f * 8)

        self.bottleneck = DoubleConv(f * 8, f * 16)

        self.dec4 = DecoderBlock(f * 16, f * 8, f * 8)
        self.dec3 = DecoderBlock(f * 8, f * 4, f * 4)
        self.dec2 = DecoderBlock(f * 4, f * 2, f * 2)
        self.dec1 = DecoderBlock(f * 2, f, f)

        self.output_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        x4, skip4 = self.enc4(x3)

        bn = self.bottleneck(x4)

        d4 = self.dec4(bn, skip4)
        d3 = self.dec3(d4, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)

        out = self.output_conv(d1)
        return out
