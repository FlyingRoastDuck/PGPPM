# -*- coding: UTF-8 -*-
from utils import *
from config import opt
import models
from dataReader import *
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm


@description
def trainModel(**kwargs):
    opt.parse(**kwargs)
    # init dataset
    dataset = dataReader(opt.imgFolder)  # fake images
    trainLoader = DataLoader(dataset, batch_size=opt.batchSize, num_workers=opt.numWorker,
                             sampler=RandomIdentitySampler(dataset.trainSet, 4), drop_last=True, pin_memory=True)
    # init model and get teacher
    model = nn.DataParallel(cpu2gpu(
        models.ResNet(modelName=opt.modelName, numCls=dataset.numID, numFea=opt.fDim)))
    # train model
    train(model, dataLoader=trainLoader, solverType='SGD', lr=opt.lr, maxEpoch=opt.maxEpoch, snap=opt.snapFreq,
          printFreq=opt.printFreq, weightDecay=opt.weightDecay, momentum=opt.momentum, lam=opt.lam,
          decayFreq=opt.decayFreq, lrDecay=opt.lrDecay, betas=[opt.beta1, opt.beta2])
    print('training done')


@description
def evaluate(**kwargs):
    opt.parse(**kwargs)
    tReader = dataReader(opt.tFolder, dataType='test', show=False)
    tLoader = DataLoader(tReader, batch_size=opt.batchSize, shuffle=False, num_workers=opt.numWorker)
    qReader = dataReader(opt.qFolder, dataType='test', show=False)
    tQLoader = DataLoader(qReader, batch_size=opt.batchSize, shuffle=False, num_workers=opt.numWorker)
    # load model
    model = nn.DataParallel(models.ResNet(modelName=opt.modelName, numCls=dataReader(opt.imgFolder).numID,
                                          numFea=opt.fDim))
    model.load_state_dict(torch.load(opt.modelSave))
    model = cpu2gpu(model)
    # eva
    eva = Evaluator(model)
    eva.evaluate(tQLoader, tLoader)


@description
def trainCamGAN(**kwargs):
    """
    train pose converter
    :param kwargs:
    :return:
    """
    opt.parse(**kwargs)
    # get visdom
    if opt.useVis:
        vis = genVisdom(opt.port)
    print('visdom active on port: {}'.format(opt.port))
    # generate training dataset
    trainSet = dataReader(opt.realTrainFolder, dataType='GAN')
    trainLoader = Data.DataLoader(trainSet, batch_size=opt.batchSize//2, shuffle=True, num_workers=opt.numWorker)
    # G,D & optimizer
    modelG = nn.DataParallel(cpu2gpu(models.Generator(camDim=trainSet.numCams)))
    modelD = nn.DataParallel(cpu2gpu(models.Discriminator(imgSize=128, camDim=trainSet.numCams)))
    optiG = torch.optim.Adam(modelG.parameters(), lr=opt.lrG, betas=[opt.beta1, opt.beta2])
    optiD = torch.optim.Adam(modelD.parameters(), lr=opt.lrD, betas=[opt.beta1, opt.beta2])
    clsLossFunc = nn.CrossEntropyLoss()
    recLoss = nn.L1Loss()
    # start training
    for ii in range(opt.maxGANEpoch):
        # update optimizer's lr
        updateGANLR(optiG, 1, ii)
        updateGANLR(optiD, 0, ii)
        print('curIter:{0:d}/{1:d}'.format(ii, opt.maxGANEpoch))
        for jj, (srcImg, _, _, cam) in enumerate(trainLoader):
            # convert 2 var
            srcData = tensor2var(cpu2gpu(srcImg))
            # camera num
            realCam = tensor2var(cpu2gpu(oneHot(cam, trainSet.numCams)))
            cam = tensor2var(cpu2gpu(cam))
            # gen fake label
            randIdx = cpu2gpu(torch.randperm(realCam.size(0)))
            fakeCamNum = cam[randIdx]
            fakeCam = tensor2var(cpu2gpu(oneHot(fakeCamNum, trainSet.numCams)))
            # train D
            realScore, realClsCam = modelD(srcData)
            lossDreal = -torch.mean(realScore)
            lossDcam = clsLossFunc(realClsCam, cam - 1)
            # distinguish from G
            fakeImg = modelG(srcData, fakeCam).detach()
            fakeScore, _ = modelD(fakeImg)
            lossDfake = torch.mean(fakeScore)
            # update D's param
            lossD = lossDcam + lossDfake + lossDreal
            optiD.zero_grad()
            lossD.backward()
            optiD.step()
            # GP term
            alpha = torch.rand(srcData.size(0), 1, 1, 1).cuda().expand_as(srcData)
            # synthesis image
            interpolated = torch.autograd.Variable(alpha * srcData.data + (1 - alpha) * srcData.data,
                                                   requires_grad=True)
            outScore, outCam = modelD(interpolated)
            grad = torch.autograd.grad(outputs=outScore,
                                       inputs=interpolated,
                                       grad_outputs=torch.ones(
                                           outScore.size()).cuda() if opt.useGpu else torch.ones(outScore.size()),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            gradL2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            lossDGP = torch.mean((gradL2norm - 1) ** 2)
            # Backward + Optimize
            lossDGP = 10 * lossDGP
            modelD.zero_grad()
            lossDGP.backward()
            optiD.step()
            # train G
            if jj % opt.GDStep == 0:
                fakeImg = modelG(srcData, fakeCam)
                recImg = modelG(fakeImg, realCam)
                #  lossG
                fakeScore, fakeCamScore = modelD(fakeImg)
                lossGfake = -torch.mean(fakeScore)
                lossGRec = recLoss(recImg, srcData)
                idtImg = modelG(srcData, realCam)
                # lossGIdt = recLoss(idtImg, srcData)
                lossGcam = clsLossFunc(fakeCamScore, fakeCamNum - 1)
                lossG = lossGfake + 10 * lossGRec + lossGcam
                optiG.zero_grad()
                lossG.backward()
                optiG.step()
                print('lossD:{0:4.2f},lossG:{1:4.2f}'.format(lossD.item(), lossG.item()))
        if opt.useVis:
            # convert cam
            bSize = srcData.shape[0]
            randIdx = cpu2gpu(torch.randperm(bSize))
            # show 3 images' results
            selectedImg = srcData[randIdx[:opt.showNum], ...]
            camID = cpu2gpu(1 + torch.from_numpy(np.arange(opt.camNum)))
            camShift = tensor2var(cpu2gpu(oneHot(camID, opt.camNum)))
            # convert cam style for each image
            genImgs = []
            for idx in range(opt.showNum):
                srcT = selectedImg[idx:idx + 1, ...]  # keep shape
                genImgs.append(modelG(srcT.expand(opt.camNum, -1, -1, -1), camShift))
            imshow(vis, genImgs, ii)
        if ii % opt.snapFreq == 0:
            torch.save(modelG.state_dict(),
                       'snapshots/G' + time.strftime('%b_%d_%H:%M:%S', time.localtime(time.time())) + '.pth')
    # save final model
    torch.save(modelG.state_dict(), opt.modelGPath)


@description
def convertCam(**kwargs):
    """
    convert images to different cam style
    """
    opt.parse(**kwargs)
    if os.path.exists(opt.imgFolder):
        print('folder already exists')
        exit(0)
    cvtSet = camReader(opt.fakeFolder)
    cvtLoader = Data.DataLoader(cvtSet, batch_size=opt.batchSize, num_workers=opt.numWorker)
    modelG = cpu2gpu(models.Generator(camDim=cvtSet.camNum))
    modelG.load(opt.modelGPath)
    modelG.train(False)
    # count images
    imgCount = np.zeros((cvtSet.imgNum, cvtSet.idNum))
    for ii, (srcImg, camLab, imgID) in enumerate(cvtLoader):
        srcData = cpu2gpu(tensor2var(srcImg))
        cvtLab = cpu2gpu(tensor2var(oneHot(1 + camLab, opt.camNum)))
        fakeImg = modelG(srcData, cvtLab)
        for jj in range(srcData.size(0)):
            saveName = str(imgID[jj]) + '_' + 'c' + str(camLab[jj] + 1) + '_' + str(
                int(imgCount[imgID[jj]][camLab[jj]])) + '.jpg'
            imgCount[imgID[jj]][camLab[jj]] += 1
            savePath = os.path.join(opt.genPath, saveName)
            save_image((1 + fakeImg[jj].data) / 2, savePath, nrow=1, padding=0)


if __name__ == '__main__':
    import fire

    fire.Fire({
        'train': trainModel,
        'test': evaluate,
        'starCam': trainCamGAN,
        'generate': convertCam
    })
