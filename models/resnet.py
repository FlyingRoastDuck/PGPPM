from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision


class ResNet(nn.Module):
    __factory = {
        'resnet18': torchvision.models.resnet18,
        'resnet34': torchvision.models.resnet34,
        'resnet50': torchvision.models.resnet50
    }

    def __init__(self, modelName, pretrained=True, numFea=0, dropout=0, numCls=0):
        super(ResNet, self).__init__()

        self.pretrained = pretrained

        # Construct base (pretrained) resnet
        if modelName not in ResNet.__factory:
            raise KeyError("Unsupported model:", modelName)
        self.model = ResNet.__factory[modelName](pretrained=pretrained)

        self.numFea = numFea
        self.dropout = dropout
        self.hasEmb = numFea > 0
        self.numCls = numCls

        inFeaDim = self.model.fc.in_features

        # Append new layers
        if self.hasEmb:
            self.feat = nn.Linear(inFeaDim, self.numFea)
            self.featBn = nn.BatchNorm1d(self.numFea)
            init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
            init.constant_(self.featBn.weight, 1)
            init.constant_(self.featBn.bias, 0)
        else:
            # Change the num_features to CNN output channels
            self.numFea = inFeaDim
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        if self.numCls > 0:
            self.classifier = nn.Linear(self.numFea, self.numCls)
            init.normal_(self.classifier.weight, std=0.001)
            init.constant_(self.classifier.bias, 0)

    def forward(self, x, outType=None):
        for name, module in self.model._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        pool5 = x

        if outType == 'pooling5':
            x = F.normalize(x)
            return x
        if self.hasEmb:
            x = self.feat(x)
            x = self.featBn(x)
        if self.hasEmb:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.numCls > 0:
            x = self.classifier(x)
        return x, pool5
