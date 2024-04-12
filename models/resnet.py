import torch
from torch import nn

from models.base import AnalogModuleBase
from modules.layers import AnalogConv2d, AnalogLayerBase, AnalogLinear


class AnalogBasicBlock(AnalogLayerBase):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, bitwidth: int = 5, leakage: float = 0.5):
        super().__init__(bitwidth=bitwidth, leakage=leakage)
        self.conv = nn.Sequential(
            AnalogConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                bitwidth=bitwidth,
                leakage=leakage,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            AnalogConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                bitwidth=bitwidth,
                leakage=leakage,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                AnalogConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    bitwidth=bitwidth,
                    leakage=leakage,
                ),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        id = self.downsample(x)
        x = self.preprocess(x, noise=False)
        id = self.preprocess(id, noise=False)
        out += id
        out = self.postprocess(out)
        out = self.relu(out)
        return out


class AnalogResNet18(AnalogModuleBase):
    def __init__(self, num_classes: int, in_channels: int = 3, bitwidth: int = 5, leakage: float = 0.5):
        super().__init__()
        self.conv0 = nn.Sequential(
            AnalogConv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, bitwidth=bitwidth, leakage=leakage
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layers = nn.Sequential(
            AnalogBasicBlock(64, 64, stride=1, bitwidth=bitwidth, leakage=leakage),
            AnalogBasicBlock(64, 64, stride=1, bitwidth=bitwidth, leakage=leakage),
            AnalogBasicBlock(64, 128, stride=2, bitwidth=bitwidth, leakage=leakage),
            AnalogBasicBlock(128, 128, stride=1, bitwidth=bitwidth, leakage=leakage),
            AnalogBasicBlock(128, 256, stride=2, bitwidth=bitwidth, leakage=leakage),
            AnalogBasicBlock(256, 256, stride=1, bitwidth=bitwidth, leakage=leakage),
            AnalogBasicBlock(256, 512, stride=2, bitwidth=bitwidth, leakage=leakage),
            AnalogBasicBlock(512, 512, stride=1, bitwidth=bitwidth, leakage=leakage),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = AnalogLinear(512, num_classes, bias=True, bitwidth=bitwidth, leakage=leakage)

    def forward(self, x: torch.Tensor):
        x = self.conv0(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
