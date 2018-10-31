# -*- coding: UTF-8 -*-
from .BasicNet import BasicNet
from .BasicNet import ResidualBlock
import torch.nn as nn
import torch


class Generator(BasicNet):
    """Generator. Encoder-Decoder Architecture."""

    def __init__(self, convDim=64, camDim=6, repeatNum=6):
        super(Generator, self).__init__()

        layers = [nn.Conv2d(3 + camDim, convDim, kernel_size=7, stride=1, padding=3, bias=False),
                  nn.InstanceNorm2d(convDim, affine=True), nn.ReLU(inplace=True)]

        # Down-Sampling
        currDim = convDim
        for i in range(2):
            layers.append(nn.Conv2d(currDim, currDim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(currDim * 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            currDim *= 2

        # Bottleneck
        for i in range(repeatNum):
            layers.append(ResidualBlock(inDim=currDim, outDim=currDim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(currDim, currDim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(currDim // 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            currDim //= 2

        layers.append(nn.Conv2d(currDim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.G = nn.Sequential(*layers)

    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.G(x)
