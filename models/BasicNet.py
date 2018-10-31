import torch.nn as nn


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.numBottleNeck = 512
        self.modelName = str(type(self)).split('\'')[-2].split('.')[-1]

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, std=0.001)
            nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    """Residual Block."""

    def __init__(self, inDim, outDim):
        super(ResidualBlock, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(inDim, outDim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(outDim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(inDim, outDim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(outDim, affine=True))

    def forward(self, x):
        return x + self.res(x)
