# -*- coding: UTF-8 -*-
from torchvision.models import *
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

supported = ['resnet18', 'resnet34', 'resnet50']


class modelFact(nn.Module):
    def __init__(self, name, outChannel, embDim=2048):
        super(modelFact, self).__init__()
        self.modelName = name  # model name
        self.embDim = embDim  # for market, 2048
        if not self.modelName.lower() in supported:
            raise NotImplementedError
        else:
            self.model = eval(name + '(pretrained=True)')
            embFdim = self.model.fc.in_features  # 2048 in default
            # some layers
            self.model.feat = nn.Linear(embFdim, embDim)
            self.model.featBN = nn.BatchNorm1d(embDim)
            nn.init.kaiming_normal_(self.model.feat.weight, mode='fan_out')
            nn.init.constant_(self.model.feat.bias, 0)
            nn.init.constant_(self.model.featBN.weight, 1)
            nn.init.constant_(self.model.featBN.bias, 0)
            # for ID prediction
            self.model.fc = nn.Linear(embDim, outChannel)
            self.model.drop = nn.Dropout(0.5)
            nn.init.normal_(self.model.fc.weight, std=0.001)
            nn.init.constant_(self.model.fc.bias, 0)

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
            out = self.model.featBN(self.model.feat(x))
            # sep 2 id and feat
            idOut = self.model.drop(self.model.fc(F.relu(out)))
            return idOut, pool5

    # def save(self, path=None):
    #     # save feature extractor is enough
    #     saveList = self.model.state_dict()
    #     if path is not None:
    #         torch.save(saveList, path)
    #     else:
    #         torch.save(saveList, 'snapshots/' + time.strftime('%b_%d_%H:%M:%S', time.localtime(time.time())) + '.pth')
    #
    # def load(self, path):
    #     if path:
    #         allParams = torch.load(path)
    #         self.model.load_state_dict(allParams)
    #     else:
    #         raise RuntimeError
