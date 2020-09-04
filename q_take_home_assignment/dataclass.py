import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import pandas as pd

class dataset(Dataset):

    def __init__(self, file):
        self.file = file
        self.readfile = file.readlines()[1:]

    def __len__(self):
        return len(self.readfile)

    def __getitem__(self, item):
        tsv = self.readfile[item].split("\t")
        return tsv


class loader:
    def __init__(self):
        self.train_data = dataset(open("glue_data/MRPC/train.tsv", encoding="UTF-8"))
        self.vad_data = dataset(open("glue_data/MRPC/dev.tsv", encoding="UTF-8"))
        self.test_data = dataset(open("glue_data/MRPC/test.tsv", encoding="UTF-8"))

    def train(self):
        return DataLoader(self.train_data, batch_size=1)

    def vad(self):
        return DataLoader(self.vad_data, batch_size=1)

    def test(self):
        return DataLoader(self.test_data, batch_size=1)

# file = open("glue_data/MRPC/train.tsv", encoding="UTF-8")
# d = dataset(file)
#
# loader = DataLoader(d, batch_size=1)
# for i in loader:
#     print(i[-2])
# print(next(iter(loader)))
# print(next(iter(loader)))
#
# for i in loader:
#     print(i)