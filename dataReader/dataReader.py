# -*- coding: UTF-8 -*-
import os
from PIL import Image
import re
import utils.transforms as T
import numpy as np
import torch
from itertools import cycle, groupby, islice
from collections import defaultdict


class dataReader(object):
    def __getitem__(self, index):
        fname = self.allF[index]
        return self.preprocess(Image.open(os.path.join(self.imgPath, fname))), fname, self.allIDs[index], self.allCams[
            index]

    def __len__(self):
        return self.len

    def __init__(self, imgPath, dataType='train', show=True):
        allF = [fname for fname in os.listdir(imgPath) if fname[-3:].lower() in ['jpg', 'png']]
        self.allF = allF
        self.imgPath = imgPath
        pat = re.compile(r'([-\d]+)_c(\d)')
        self.len = len(allF)
        allIDs, allCams = [], []
        for ii in range(self.len):
            curID, curCams = map(int, pat.search(allF[ii]).groups())
            allIDs.append(curID)
            allCams.append(curCams)
        self.allIDs = allIDs
        self.allCams = allCams
        self.numID = len(set(allIDs))
        self.numCams = len(set(allCams))
        self.preprocess = None
        self.trainSet = []
        imgSize = [128, 128]
        [self.trainSet.append([allF[i], allIDs[i], allCams[i]]) for i in range(len(allF))]
        # throw info
        if dataType == 'train':
            self.preprocess = T.Compose([
                T.RandomSizedRectCrop(256, 128),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            if show:
                print(
                    '-----* Training with {total} images ({numID} classes) *-----'.format(total=self.len,
                                                                                          numID=self.numID))
        elif dataType == 'test':
            self.preprocess = T.Compose([
                T.RectScale(256, 128),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif dataType == 'GAN':
            self.preprocess = T.Compose([
                T.Resize((int(1.125 * imgSize[0]), int(1.125 * imgSize[1]))),
                T.RandomCrop(imgSize),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class knnCameraReader(object):
    def __init__(self, dataset, transform=None, knnIndex=None, distance=None):
        self.dataset = dataset.trainSet  # sorted
        self.root = dataset.imgPath
        self.transform = transform
        self.knnIndex = knnIndex
        self.distance = distance  # neighbour similarity

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self.__getSingleItem(index) for index in indices]
        return self.__getSingleItem(indices)

    def __getWeight(self, index, kRe):
        disExp = self.distance[index, kRe].numpy()  # k-reciprocal dist
        weightDis = np.asarray([1e-16] * len(kRe))
        for ii in range(len(kRe)):
            curKre = kRe[ii]
            chosenRe = self.knnIndex[curKre]  # k-re's k-reciprocal,self not included
            chosenReDis = np.exp(-self.distance[curKre, chosenRe].numpy())  # k-re's k-reciprocal dist
            mask = np.asarray([x in chosenRe for x in kRe])  # in curKre's kRe? do not forget itself
            maskNN = np.asarray([x in kRe for x in chosenRe])  # in chosenRe's kRe?
            if True in mask:  # in mask will leads to in maskNN
                normW = chosenReDis[maskNN] / chosenReDis[maskNN].sum()
                weightDis[ii] = np.exp(-disExp[mask]).dot(normW)  # weigted
        weightDis += np.exp(-disExp)
        return weightDis.tolist()

    def __getSingleItem(self, index):
        allImg = []
        allFname = []
        allPid = []
        allCam = []
        fname, pid, camid = self.dataset[index]
        pid = int(torch.rand(1) * 10000 + 10000)  # fake label
        if self.root is not None:
            fpath = os.path.join(self.root, fname)

            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            allImg.append(img)
            allFname.append(fname)
            allPid.append(pid)
            allCam.append(camid)

        if self.knnIndex is not None:
            curKnnIndex = self.knnIndex[index]  # query's k-re
            curKnnIndex = list(set(curKnnIndex) - set([index]))
            if len(self.knnIndex[index]) > 1:
                # compute weight for all k-re
                curWeight = self.__getWeight(index, curKnnIndex)
                # samples with max weight
                curWeight = np.asarray(curWeight)
                maxLoc = curWeight.argmax()

                # for anay
                # sortLoc = curWeight.argsort()[::-1] # loc in desending order
                # kreIndex = np.asarray(curKnnIndex)
                # sortedIndex = kreIndex[sortLoc]
                # allKreName=""
                # # get all names
                # for ii in range(len(sortedIndex)):
                #     valName,_,_ = self.dataset[sortedIndex[ii]]
                #     allKreName=allKreName+valName+","
                # allKreName+=fname
                # allKreName+="\n"

                maxName, _, _ = self.dataset[curKnnIndex[maxLoc]]

                fpath = os.path.join(self.root, maxName)
                allImg.append(self.transform(Image.open(fpath).convert('RGB')))
                allFname.append(maxName)
                allPid.append(pid)
                allCam.append(camid)

                # with open('highWMar.txt','a+') as fs:
                #     # fs.write('{0:},{1:}\n'.format(fname, maxName))
                #     fs.write(allKreName)
            else:
                return allImg, allFname, allPid, allCam

        return allImg, allFname, allPid, allCam


def groupByID(nameList):
    results = defaultdict(list)
    allItems = groupby(nameList, key=lambda x: int(x.split('_')[0]))
    for ID, name in allItems:
        results[ID].append(list(name))
    return results


class camReader(object):
    def __init__(self, srcFolder, transformation=None):
        # get name
        allF = os.listdir(srcFolder)
        imgSize = [128, 128]
        groupedName = groupByID(allF)
        # transformation of srcImg
        if transformation is None:
            self.trans = T.Compose([
                T.Resize(imgSize),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.trans = transformation
        # labeler
        pat = re.compile(u'([-\d]+)_c(\d)')
        allCams = []
        for ii in range(len(allF)):
            _, curCams = map(int, pat.search(allF[ii]).groups())
            allCams.append(curCams)
        self.camNum = len(set(allCams))
        camLaber = cycle(range(self.camNum))
        # label each fake image with camera
        fName, tarCam = [], []
        for ID, names in groupedName.items():
            # convert names to single list
            curRes = []
            for ii in range(len(names)): curRes += names[ii]
            fName += curRes
            tarCam += list(islice(camLaber, 0, len(curRes), 1))
        self.allName = fName
        self.tarCam = tarCam
        self.imgNum = len(self.allName)
        self.srcFolder = srcFolder
        # number of ID
        self.idNum = max(list(map(lambda x: int(x.split('_')[0]), allF))) + 1

    def __getitem__(self, index):
        imgName = self.allName[index]  # ori name
        srcImg = self.trans(Image.open(os.path.join(self.srcFolder, imgName)))
        ID = int(imgName.split('_')[0])
        return srcImg, self.tarCam[index], ID

    def __len__(self):
        return self.imgNum
