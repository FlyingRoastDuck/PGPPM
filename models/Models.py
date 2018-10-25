# -*- coding: UTF-8 -*-
import torchvision.models as tModels
import torch.nn as nn
from torch.nn import functional as F

supported = {
    'resnet18': tModels.resnet50,
    'resnet34': tModels.resnet34,
    'resnet50': tModels.resnet50
}


class modelFact(nn.Module):
    def __init__(self, name, outChannel, embDim=1024):
        super(modelFact, self).__init__()
        self.modelName = name  # model name
        self.embDim = embDim  # for market, 2048
        if not self.modelName.lower() in supported.keys():
            raise NotImplementedError
        else:
            self.model = supported[name](pretrained=True)
            embFdim = self.model.fc.in_features  # 2048 in default
            # some layers
            self.feat = nn.Linear(embFdim, embDim)
            self.featBN = nn.BatchNorm1d(embDim)
            nn.init.kaiming_normal_(self.feat.weight, mode='fan_out')
            nn.init.constant_(self.feat.bias, 0)
            nn.init.constant_(self.featBN.weight, 1)
            nn.init.constant_(self.featBN.bias, 0)
            # for ID prediction
            self.fc = nn.Linear(embDim, outChannel)
            self.drop = nn.Dropout(0.5)
            nn.init.normal_(self.fc.weight, std=0.001)
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, outType='normal'):
        # forward 2 avgpool
        for keys, module in self.model._modules.items():
            if keys == 'avgpool':
                break
            x = module(x)
        # GMP
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        pool5 = x
        if outType == 'pooling5':
            return F.normalize(x)  # use normalized whole vec to eva
        else:
            out = self.featBN(self.feat(x))
            # sep 2 id and feat
            idOut = self.fc(self.drop(F.relu(out)))
            return idOut, pool5
