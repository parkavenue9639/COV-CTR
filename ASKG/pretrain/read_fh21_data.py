import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import os
import pickle as pkl
import json
import numpy as np
import sys


class Fh21Dataset(Dataset):
    def __init__(self, opt):
        
        # TODO
        self.data_dir = '../report/processed_fh21_precise_tag/'
        self.num_medterm = opt.num_medterm  # 指定医学术语的数量

        # 读取pkl文件，该文件是一个pickel序列化的文件，存储了数据集的主要内容
        with open(os.path.join(self.data_dir, 'fh21.pkl'), 'rb') as f:
            self.data = pkl.load(f)

        # 该文件是一个word-to-id词典，将单词映射到唯一的整数id
        with open(os.path.join(self.data_dir, 'word2idw.pkl'), 'rb') as f:
            self.word2idw = pkl.load(f)

        # 创建反向词典，用于将id转换为单词
        self.idw2word = {v: k for k, v in self.word2idw.items()}

        # 获取词汇大小
        self.vocab_size = len(self.word2idw)

        print("Fh21Dataset init complete")

    def __getitem__(self, index):
        # 用于获取index位置的数据，返回索引、摘要数据、摘要标签、医学术语标签
        ix = self.data[index][0]
        abstracts = self.data[index][1]
        abstracts = np.array(abstracts)

        abstracts_labels = self.data[index][2]
        abstracts_labels = np.array(abstracts_labels)

        # 转换为torch张量
        abstracts = torch.from_numpy(abstracts).long()
        abstracts_labels = torch.from_numpy(abstracts_labels).long()

        # 处理医学术语标签
        medterm_labels = np.zeros(229)
        medterms = self.data[index][3]
        for medterm in medterms:
            # medterm_labels[medterm] = 1
            if medterm < self.num_medterm:
                # # 独热向量编码向量，表示该摘要包含哪些医学术语。
                medterm_labels[medterm] = 1  # 表示该摘要设计medterm这个医学术语

        # 数据索引、转换为torch张量的摘要、摘要的标签、医学术语的独热编码量
        return ix, abstracts, abstracts_labels, torch.FloatTensor(medterm_labels)

    def __len__(self):
        #  数据集长度
        return len(self.data)


def get_loader2(opt):

    dataset = Fh21Dataset(opt)
    loader = DataLoader(dataset=dataset, batch_size=opt.train_batch_size,
                            shuffle=True, num_workers=16)
    return dataset, loader

