from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    def __init__(self, dataSource, numInstances=1):
        self.dataSource = dataSource
        self.numInstances = numInstances
        self.indexDic = defaultdict(list)
        for index, (_, pid, _) in enumerate(dataSource):
            self.indexDic[pid].append(index)
        self.pids = list(self.indexDic.keys())
        self.numSamples = len(self.pids)

    def __len__(self):
        return self.numSamples * self.numInstances

    def __iter__(self):
        indices = torch.randperm(self.numSamples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.indexDic[pid]
            if len(t) >= self.numInstances:
                t = np.random.choice(t, size=self.numInstances, replace=False)
            else:
                t = np.random.choice(t, size=self.numInstances, replace=True)
            ret.extend(t)
        return iter(ret)
