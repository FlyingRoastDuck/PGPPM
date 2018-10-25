# -*- coding: UTF-8 -*-
import numpy as np
import torch.optim as Optim
from torch.autograd import Variable
import time
from .evaluationMetrics import cmc, mean_ap
from collections import OrderedDict
import sys
from loss import TripletLoss
from dataReader import *
import torch.utils.data as Data

sys.path.append('../')

from config import opt

allFuncDes = {
    'trainModel': '-----* training model with source data *-----',
    'evaluate': '-----* Test Model Adaptation *-----'
}


def getSolver(solverType, model, **kwargs):
    """
    solver type
    :param solverType:
    :param model:
    :param kwargs:
    :return:
    """
    if solverType.lower() == 'adam':
        print('-----* using adam solver *-----')
        optimizer = Optim.Adam(model.parameters(), lr=kwargs['lr'], betas=kwargs['betas'])
    elif solverType.lower() == 'sgd':
        optimizer = Optim.SGD(model.parameters(), lr=kwargs['lr'], momentum=kwargs['momentum'],
                              weight_decay=kwargs['weightDecay'], nesterov=True)
    else:
        raise NotImplementedError
    return optimizer


def updateLR(optimizer, epoch, solverType='SGD', **kwargs):
    """
    update learning rate according to solverType
    :param optimizer:
    :param epoch:
    :param solverType:
    :param kwargs:
    :return:
    """
    if solverType.lower() == 'adam':
        if (epoch - int(0.8 * kwargs['maxEpoch'])) > 0:
            lr = opt.lr
            lr -= lr / float(kwargs['maxEpoch'] - int(0.8 * kwargs['maxEpoch']))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print("lr declined")
    elif solverType.lower() == 'sgd':
        curLR = opt.lr * (kwargs['lrDecay'] ** (epoch // opt.startTrip))
        for g in optimizer.param_groups:
            g['lr'] = curLR
    else:
        raise NotImplementedError


def train(model, dataLoader, solverType='SGD', **kwargs):
    """
    train model with real and fake
    :param model:
    :param dataLoader:
    :param solverType:
    :param kwargs:
    :return:
    """
    optimizer = getSolver(solverType, model, lr=kwargs['lr'], momentum=kwargs['momentum'],
                          weightDecay=kwargs['weightDecay'], betas=kwargs['betas'])
    # losses
    idLoss = torch.nn.CrossEntropyLoss()
    triLoss = TripletLoss(opt.margin)
    model = model.train(True)
    for ii in range(kwargs['maxEpoch']):
        updateLR(optimizer, ii, solverType=solverType, decayFreq=kwargs['decayFreq'], maxEpoch=kwargs['decayFreq'],
                 lr=kwargs['lr'], lrDecay=kwargs['lrDecay'])
        if ii < opt.startTrip:
            for jj, (imgData, _, pid, _) in enumerate(dataLoader):
                startT = time.time()
                imgData = cpu2gpu(tensor2var(imgData))
                pid = cpu2gpu(tensor2var(pid))
                idScore, _ = model(imgData)
                lossCls = idLoss(idScore, pid)
                optimizer.zero_grad()
                loss = lossCls
                loss.backward()
                optimizer.step()
                endT = time.time()
                if jj % kwargs['printFreq'] == 0:
                    print(
                        'epoch:[{epoch}/{maxEpoch}],Round:[{curJ}/{allJ}], id loss:{idloss}, elapsed time:{time}s'.format(
                            epoch=ii, maxEpoch=kwargs['maxEpoch'], curJ=jj, allJ=len(dataLoader),
                            idloss=lossCls, time=endT - startT))
        else:
            knnLoader = getKNNLoader(opt.realTrainFolder, model, numPerson=opt.triBatch, K=opt.K)
            for (jj, (imgData, _, pid, _)), (knnImg, _, pidKNN, _) in zip(enumerate(dataLoader), knnLoader):
                startT = time.time()
                knnImg = cpu2gpu(tensor2var(torch.cat(knnImg)))
                pidKNN = cpu2gpu(tensor2var(torch.cat(pidKNN)))
                imgData = cpu2gpu(tensor2var(imgData))
                pid = cpu2gpu(tensor2var(pid))
                idScore, _ = model(imgData)
                _, tripEmb = model(knnImg)
                # CAL LOSS
                lossCls = idLoss(idScore, pid)
                lossTri, precTri = triLoss(tripEmb, pidKNN)
                # update
                optimizer.zero_grad()
                loss = lossCls + kwargs['lam'] * lossTri
                loss.backward()
                optimizer.step()
                endT = time.time()
                if jj % kwargs['printFreq'] == 0:
                    print(
                        'epoch:{epoch}/{maxEpoch},Round:[{curJ}/{allJ}], id loss:{idloss}, triplet loss:{triL}, elapsed time:{time}s'.format(
                            epoch=ii,
                            maxEpoch=kwargs['maxEpoch'],
                            curJ=jj,
                            allJ=len(dataLoader),
                            idloss=lossCls,
                            triL=lossTri, time=endT - startT))
        print('Epoch {} Done !'.format(ii))
        if ii % kwargs['snap'] == 0:
            # model.save()  # save snaps
            torch.save(model.state_dict(),
                       'snapshots/' + time.strftime('%b_%d_%H:%M:%S', time.localtime(time.time())) + '.pth')
            print('snapshots saved!')
    # return new model
    torch.save(model.state_dict(), opt.modelSave)


def extractCNNfeature(dataLoader, model):
    model = model.train(False)
    feat = np.zeros([dataLoader.dataset.len, 2048])  # pooling5 for comparison
    allFnames = []
    index = 0
    allCams, allIDs = np.zeros([dataLoader.dataset.len, 1]), np.zeros([dataLoader.dataset.len, 1])
    for imgData, fnames, pid, cam in dataLoader:
        startT = time.time()
        imgData = cpu2gpu(tensor2var(imgData))
        Feat = model(imgData, outType='pooling5')
        for (curFeat, curName, curPid, curCam) in zip(Feat, fnames, pid, cam):
            feat[index] = gpu2cpu(var2tensor(curFeat)).numpy()
            allFnames.append(curName)
            allCams[index], allIDs[index] = gpu2cpu(var2tensor(curCam)).numpy(), gpu2cpu(var2tensor(curPid)).numpy()
            index += 1
        endT = time.time()
        print('Extracting Features... Time:{time}s'.format(time=endT - startT))
    print('Done!')
    return allFnames, allIDs, allCams, feat


def getKNNLoader(dstImgPath, model, **kwargs):
    K, numPerson = kwargs["K"], kwargs["numPerson"]
    featureSet = dataReader(dstImgPath, dataType='train', show=False)  # real image path
    dataSet = dataReader(dstImgPath, dataType='test', show=False)  # for feature extraction
    # get loader
    extractLoader = Data.DataLoader(dataSet, batch_size=128, num_workers=opt.numWorker, pin_memory=True, shuffle=False)
    # geiFeatures
    allFnames, allIDs, allCams, feat = extractCNNfeature(extractLoader, model)
    allKNN, disMat = knnIndex(feat, K, allCams)
    print('Mining Done')
    knnLoader = Data.DataLoader(
        knnCameraReader(featureSet, transform=featureSet.preprocess, knnIndex=allKNN, distance=disMat),
        batch_size=numPerson, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
    return knnLoader


def knnIndex(feat, K, camLabs):
    feat = torch.from_numpy(feat)
    allNum = feat.shape[0]
    dist = pairwiseDis(feat, feat)
    # let same cam to be large
    camLabs = torch.from_numpy(camLabs).squeeze()
    numDim = camLabs.shape[0]
    camX, camY = camLabs.unsqueeze(1), camLabs.unsqueeze(0)
    camX, camY = camX.expand((numDim, numDim)), camY.expand((numDim, numDim))
    mask = camX.eq(camY)
    calDis = dist.clone()
    calDis[mask] = 100  # "big value", remove same camera
    del feat
    initRank = np.argpartition(calDis.cpu().numpy(), range(1, K + 3))
    allKNN = []
    for i in range(allNum):
        # k-reciprocal neighbors
        fkIndex = initRank[i, :K + 1]
        bkIndex = initRank[fkIndex, :K + 1]
        fi = np.where(bkIndex == i)[0]
        kreIndex = fkIndex[fi]

        kExpandIndex = np.unique(kreIndex)
        allKNN.append(kExpandIndex)
    return allKNN, calDis


def extractF(dataLoader, model):
    """
    extract pooling 5 features for dataLoader
    :param imgData:
    :param model:
    :return:
    """
    model = model.train(False)
    feature = OrderedDict()
    for ii, (imgData, fname, pid, cams) in enumerate(dataLoader):
        imgData = cpu2gpu(tensor2var(imgData))
        feats = model(imgData, outType='pooling5')
        for feat, name in zip(gpu2cpu(var2tensor(feats)), fname):
            feature[name] = feat
    print('-----* Feature Obtained *-----')
    return feature


def pairwiseDis(qFeature, gFeature):
    """
    com pair wise distance
    :param qFeature:
    :param gFeature:
    :return:
    """
    x, y = qFeature, gFeature
    m, n = x.shape[0], y.shape[0]
    disMat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    disMat.addmm_(1, -2, x, y.t())
    print('-----* Distance Matrix has been computed*-----')
    return disMat.clamp_(min=1e-12)


def evaModel(disMat, qPID, gPID, qCam, gCam, cmcTop=(1, 5, 10, 20)):
    mAP = mean_ap(disMat, qPID, gPID, qCam, gCam)
    print('Mean AP: {:4.1%}'.format(mAP))
    # cal CMC
    cmcConfigs = dict(separate_camera_set=False,
                      single_gallery_shot=False,
                      first_match_break=True)
    cmcScores = cmc(disMat, qPID, gPID, qCam, gCam, **cmcConfigs)
    print('CMC Scores:')
    for k in cmcTop:
        print('  top-{:<4}{:12.1%}'.format(k, cmcScores[k - 1]))

    return cmcScores[0]


def calDisForEva(qFeat, gFeat):
    x, y = torch.cat([qFeat[f].unsqueeze(0) for f in qFeat.keys()], 0), torch.cat([gFeat[f].unsqueeze(0) for f in gFeat.keys()], 0)
    m, n = x.shape[0], y.shape[0]
    disMat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    disMat.addmm_(1, -2, x, y.t())
    print('Distance Matrix for Evaluation has been computed')
    return disMat.clamp_(min=1e-12)


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model
        self.pat = re.compile(r'([-\d]+)_c(\d)')

    def evaluate(self, qLoader, gLoader):
        qFeat = extractF(qLoader, self.model)
        gFeat = extractF(gLoader, self.model)
        # getP PID and Cam
        qPID = [int(pid) for _, pid, _ in qLoader.trainSet]
        qCAM = [int(cam) for _, _, cam in qLoader.trainSet]
        gPID = [int(pid) for _, pid, _ in gLoader.trainSet]
        gCAM = [int(cam) for _, _, cam in gLoader.trainSet]
        disMat = calDisForEva(qFeat, gFeat)
        import ipdb;ipdb.set_trace()
        return evaModel(disMat, qPID=qPID, gPID=gPID, qCam=qCAM, gCam=gCAM)


def var2tensor(var):
    return var.data


def tensor2var(val, grad=False):
    return Variable(val, requires_grad=grad)


def cpu2gpu(var):
    return var.cuda() if opt.useGpu else var


def gpu2cpu(var):
    return var.cpu() if var.is_cuda else var


def description(func):
    """
    decorator for main funcs
    :param func:
    :return:
    """

    def warpper(**kwargs):
        if func.__name__ in allFuncDes.keys():
            print(allFuncDes[func.__name__])
        else:
            raise NotImplementedError
        func(**kwargs)

    return warpper
