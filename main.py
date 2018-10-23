# -*- coding: UTF-8 -*-
from utils import *
from config import opt
import models
from dataReader import *
from torch.utils.data import DataLoader
import torch.nn as nn


@description
def trainModel(**kwargs):
    opt.parse(**kwargs)
    # init dataset
    dataset = dataReader(opt.imgFolder)  # fake images
    trainLoader = DataLoader(dataset, batch_size=opt.batchSize, num_workers=opt.numWorker,
                             sampler=RandomIdentitySampler(dataset.trainSet, 4), drop_last=True, pin_memory=True)
    # init model and get teacher
    model = nn.DataParallel(cpu2gpu(
        models.modelFact(name=opt.modelName, isInit=opt.pretrain, outChannel=dataset.numID, embDim=opt.fDim)))
    # train model
    model = train(model, dataLoader=trainLoader, solverType='SGD', lr=opt.lr, maxEpoch=opt.maxEpoch, snap=opt.snapFreq,
                  printFreq=opt.printFreq, weightDecay=opt.weightDecay, momentum=opt.momentum, lam=opt.lam,
                  decayFreq=opt.decayFreq, lrDecay=opt.lrDecay, betas=[opt.beta1, opt.beta2])
    # model.save(opt.modelSave)
    torch.save(model.state_dict(), opt.modelSave)
    print('training done')


@description
def evaluate(**kwargs):
    opt.parse(**kwargs)
    tReader = dataReader(opt.tFolder, dataType='test')
    tLoader = DataLoader(tReader, batch_size=opt.batchSize, shuffle=True, num_workers=opt.numWorker)
    qReader = dataReader(opt.qFolder, dataType='test')
    tQLoader = DataLoader(qReader, batch_size=opt.batchSize, shuffle=True, num_workers=opt.numWorker)
    # load model
    model = models.modelFact(name=opt.modelName, isInit=opt.pretrain, outChannel=dataReader(opt.imgFolder).numID,
                             embDim=opt.fDim)
    model.load(opt.modelSave)
    model = cpu2gpu(model)
    # eva
    eva = Evaluator(model)
    eva.evaluate(tQLoader, tLoader)


if __name__ == '__main__':
    import fire

    fire.Fire({
        'train': trainModel,
        'test': evaluate,
    })
