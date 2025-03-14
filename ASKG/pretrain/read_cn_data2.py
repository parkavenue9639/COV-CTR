import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from PIL import Image

import os
import pickle as pkl
import json
import numpy as np
import sys


class CHXrayDataSet2(Dataset):
    def __init__(self, opt, split, transform=None):
        self.transform = transform

        self.data_dir = opt.data_dir
        # TODO
        self.pkl_dir = os.path.join('..', 'report')
        self.img_dir = os.path.join(self.data_dir, 'NLMCXR_png')

        self.num_medterm = opt.num_medterm  # 指定医学术语的数量

        with open(os.path.join(self.pkl_dir, 'align2.' + split + '.pkl'), 'rb') as f:
            self.findings = pkl.load(f)  # 影像对应的文本描述
            self.findings_labels = pkl.load(f)  # 文本描述的标签
            self.image = pkl.load(f)  # 影像文件名
            self.medterms = pkl.load(f)  # 影像对应的医学术语  length:1470

        f.close()

        with open(os.path.join(self.pkl_dir, 'word2idw.pkl'), 'rb') as f:
            self.word2idw = pkl.load(f)  # 单词到id的映射，用于文本处理
        f.close()

        with open(os.path.join(self.pkl_dir, 'idw2word.pkl'), 'rb') as f:
            self.idw2word = pkl.load(f)  # id到单词的映射，用于解码文本
        f.close()

        self.ids = list(self.image.keys())  # 获取所有的影像id
        self.vocab_size = len(self.word2idw)  # 词汇表大小

        print('CHXrayDataSet2 init complete')

    def __getitem__(self, index):
        ix = self.ids[index]  # 获取影像id
        image_id = self.image[ix]  # 获取影像文件名
        image_name = os.path.join(self.img_dir, image_id)  # 拼接完整路径
        img = Image.open(image_name).convert('RGB')  # 打开图像，转换为RGB（避免PIL处理灰度图时出错）
        if self.transform is not None:
            img = self.transform(img)  # 数据增强，对图像进行变换

        #print(img.size(), image_id)
        medterm_labels = np.zeros(229)
        # medterm_labels = np.zeros(self.num_medterm)
        medterms = self.medterms[ix]
        for medterm in medterms:
            # medterm_labels[medterm] = 1
            if medterm < self.num_medterm:
                medterm_labels[medterm] = 1
        # print("medterm_labels shape{}".format(medterm_labels.shape))  # (50176,)

        findings = self.findings[ix]
        findings_labels = self.findings_labels[ix]
        findings = np.array(findings)
        findings_labels = np.array(findings_labels)

        findings = torch.from_numpy(findings).long()
        findings_labels = torch.from_numpy(findings_labels).long()


        return ix, image_id, img, findings, findings_labels, torch.FloatTensor(medterm_labels)

    def __len__(self):
        return len(self.ids)


def get_loader_cn(opt, split):

    # 采用ImageNet的均值和标准差，这样可以加速模型收敛，因为常见的预训练模型都是用ImageNet数据集训练的，保持相同的归一化方式有助于更好的迁移学习
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    # 数据集封装
    dataset = CHXrayDataSet2(opt, split=split,  # split指定数据的类别
                             transform=transforms.Compose([
                                 transforms.Resize((224, 224)),  # 将图像调整到224x224，适配CNN结构
                                 transforms.ToTensor(),  # 将图像转换为[C H W]格式的tensor，并归一化到[0,1]
                                 normalize  # 归一化方式
                             ]))
    if split == 'train':
        loader = DataLoader(dataset=dataset, batch_size=opt.train_batch_size,
                            shuffle=True, num_workers=16)
    elif split == 'val':
        loader = DataLoader(dataset=dataset, batch_size=opt.eval_batch_size,
                            shuffle=True, num_workers=16)
    elif split == 'test':
        loader = DataLoader(dataset=dataset, batch_size=opt.eval_batch_size,
                            shuffle=False, num_workers=16)
    else:
        raise Exception('DataLoader split must be train or val.')
    return dataset, loader

