"""
ResNet34 當編碼器、UNet 風格當解碼器的二元分割網路。

參考：
    ResNet：https://arxiv.org/abs/1512.03385
    類似編碼—解碼示意：https://www.researchgate.net/publication/359463249

encoder 會拉出多尺度特徵給 skip；decoder 逐步放大並融合這些特徵。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- ResNet34 編碼端 ---


class BasicBlock(nn.Module):
    """ResNet 的 BasicBlock（ResNet-18／34 都用這個）。"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def _make_layer(in_channels, out_channels, num_blocks, stride=1):
    """堆疊一串 BasicBlock，必要時用 1×1 conv 調整維度。"""
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    layers = [
        BasicBlock(in_channels, out_channels, stride=stride, downsample=downsample)
    ]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class ResNet34Encoder(nn.Module):
    """
    ResNet-34 前半（不含全域 pool 與分類全連接）。
    forward 會回傳各階段的 feature map，給 decoder 當 skip。
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-34：每段 block 數為 3, 4, 6, 3
        self.layer1 = _make_layer(64, 64, num_blocks=3, stride=1)
        self.layer2 = _make_layer(64, 128, num_blocks=4, stride=2)
        self.layer3 = _make_layer(128, 256, num_blocks=6, stride=2)
        self.layer4 = _make_layer(256, 512, num_blocks=3, stride=2)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.maxpool(s0)
        s1 = self.layer1(s1)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return s0, s1, s2, s3, s4


# --- UNet 風格解碼端 ---


class ConvBnRelu(nn.Module):
    """單層 Conv-BN-ReLU。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetDecoderBlock(nn.Module):
    """解碼一小段：上採樣 → concat skip → 兩層卷積。"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvBnRelu(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBnRelu(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# --- 完整 ResNet34 + UNet ---


class ResNet34UNet(nn.Module):
    """
    ResNet34 編碼 + UNet 式解碼（二元輸出 logits）。
    最後再上採樣一次並用 1×1 conv 得到單通道遮罩。
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels=in_channels)

        self.dec3 = UNetDecoderBlock(512, 256, 256)
        self.dec2 = UNetDecoderBlock(256, 128, 128)
        self.dec1 = UNetDecoderBlock(128, 64, 64)
        self.dec0 = UNetDecoderBlock(64, 64, 32)

        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.output_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        s0, s1, s2, s3, s4 = self.encoder(x)

        d3 = self.dec3(s4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)

        out = self.final_up(d0)
        out = self.output_conv(out)
        return out
