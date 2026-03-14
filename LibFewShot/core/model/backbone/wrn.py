# -*- coding: utf-8 -*-
"""
Modified wrn.py:
Structure aligned with WideResNet.py, but kept strictly as a feature extractor.
"""
import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropRate=0.0, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropRate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        # Align shortcut connection logic with WideResNet.py
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        # Aligned forward pass: BN -> ReLU -> Conv -> Dropout -> BN -> ReLU -> Conv
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(
            self, depth, widen_factor=1, dropRate=0.0, is_flatten=True, avg_pool=True
    ):
        super(WideResNet, self).__init__()
        # Keep original feature extractor flags
        self.is_flatten = is_flatten
        self.avg_pool = avg_pool

        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6

        nStages = [
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
        ]

        # Track input channels dynamically like WideResNet.py
        self.in_planes = 16

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nStages[0], kernel_size=3, stride=1, padding=1, bias=True
        )

        # Build network blocks using _wide_layer
        self.layer1 = self._wide_layer(BasicBlock, nStages[1], n, dropRate, stride=1)
        self.layer2 = self._wide_layer(BasicBlock, nStages[2], n, dropRate, stride=2)
        self.layer3 = self._wide_layer(BasicBlock, nStages[3], n, dropRate, stride=2)

        # global average pooling and classifier preparation
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nStages[3]

        # Aligned Initialization (Kaiming Normal)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _wide_layer(self, block, planes, num_blocks, dropRate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, dropRate, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn1(out))

        # Retain original wrn.py output logic for feature extraction
        if self.avg_pool:
            out = F.adaptive_avg_pool2d(out, 1)
        if self.is_flatten:
            out = out.reshape(out.size(0), -1)

        return out


def WRN(**kwargs):
    """
    Constructs a Wide Residual Networks (Feature Extractor).
    """
    model = WideResNet(**kwargs)
    return model