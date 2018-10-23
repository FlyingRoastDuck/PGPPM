# -*- coding: UTF-8 -*-
import os
from PIL import Image
import re
import torchvision.transforms as T
import numpy as np
import torch


class dataReader(object):
    def __getitem__(self, index):
        fname = self.allF[index]
        return self.preprocess(Image.open(os.path.join(self.imgPath, fname))), fname, self.allIDs[index], self.allCams[index]

    def __len__(self):
        return self.len

    def __init__(self, imgPath, dataType='train'):
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
        self.numID = len(np.unique(allIDs))
        self.preprocess = None
        self.trainSet = []
        [self.trainSet.append([allF[i], allIDs[i], allCams[i]]) for i in range(len(allF))]
        # throw info
        if dataType == 'train':
            self.preprocess = T.Compose([
                T.Resize([256, 128]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print(
                '-----* Training with {total} images ({numID} classes) *-----'.format(total=self.len, numID=self.numID))
        elif dataType == 'test':
            self.preprocess = T.Compose([
                T.Resize([256, 128]),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print('-----* Testing with {total} images *-----'.format(total=self.len))


class knnCameraReader(object):
    def __init__(self, dataset, transform=None, knnIndex=None, distance=None):
        allCams, allPids, allF = dataset.allCams, dataset.allIDs, dataset.allF
        self.dataset = dataset.trainSet
        self.root = dataset.imgPath
        self.transform = transform
        self.knn_index = knnIndex
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
            chosenRe = self.knn_index[curKre]  # k-re's k-reciprocal,self not included
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
        pid = int(torch.rand(1) * 10000 + 10000)
        if self.root is not None:
            fpath = os.path.join(self.root, fname)

            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            allImg.append(img)
            allFname.append(fname)
            allPid.append(pid)
            allCam.append(camid)

        if self.knn_index is not None:
            curKnnIndex = self.knn_index[index]  # query's k-re
            curKnnIndex = list(set(curKnnIndex) - set([index]))
            if len(self.knn_index[index]) > 1:
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

                fpath = os.path.join(self.root, self.path_name, maxName)
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
