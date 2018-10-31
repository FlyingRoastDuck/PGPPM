# -*- coding: UTF-8 -*-
import torch.nn as nn
import numpy as np
from .BasicNet import BasicNet


class Discriminator(BasicNet):
    """Discriminator. PatchGAN."""

    def __init__(self, imgSize=128, convDim=64, camDim=5, repeatNum=6):
        super(Discriminator, self).__init__()
        layers = [nn.Conv2d(3, convDim, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.01, inplace=True)]
        currDim = convDim
        for i in range(1, repeatNum):
            layers.append(nn.Conv2d(currDim, currDim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            currDim *= 2
        kSize = int(imgSize / np.power(2, repeatNum))
        self.D = nn.Sequential(*layers)
        # score of a real image
        self.conv1 = nn.Conv2d(currDim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # cam classification
        self.conv2 = nn.Conv2d(currDim, camDim, kernel_size=kSize, bias=False)
        # # person classification
        # self.conv3 = nn.Conv2d(currDim, numCls, kernel_size=kSize, bias=False)

    def forward(self, x):
        # return realConfident,camClass,personClass
        h = self.D(x)
        realOut = self.conv1(h)
        camOut = self.conv2(h)
        # outID = self.conv3(h)
        return realOut.squeeze(), camOut.squeeze()  # , outID.squeeze()
