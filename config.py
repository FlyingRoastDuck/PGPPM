# -*- coding: UTF-8 -*-
import warnings


class Config(object):
    def __init__(self):
        self.imgFolder = '/home/yfx/zzdEval/market/camVM60036'  # marketORIpath to images
        self.realTrainFolder = '/home/yfx/zzdEval/market/train'
        self.qFolder = '/home/yfx/zzdEval/market/query'  # FOR DEBUG
        self.tFolder = '/home/yfx/zzdEval/market/test'
        self.modelSave = 'snapshots/PGPPM.pth'
        # basic param for model
        self.modelName = 'resnet50'
        self.lr = 0.01
        self.margin = 0.3
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.momentum = 0.9
        self.weightDecay = 5e-4
        self.fDim = 1024
        self.batchSize = 60
        self.triBatch = 30
        self.K = 50
        self.maxEpoch = 150
        self.startTrip = 100
        self.decayFreq = 20
        self.lrDecay = 0.1  # lr decay
        self.lam = 1  # GP term
        self.useGpu = True  # gpu mode
        self.numWorker = 8  # thread
        self.snapFreq = 60  # save freq
        self.printFreq = 1

    def parse(self, **kwargs):
        # user's config
        for k, v in kwargs.items():
            if not getattr(self, k):
                warnings.warn("no %s found" % k)
            setattr(self, k, v)
        print('user config:')
        for k, v in vars(self).items():
            if not k.startswith('__'):
                print(k, v)


opt = Config()  # a config class
