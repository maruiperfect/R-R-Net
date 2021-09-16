# -*- coding: utf-8 -*-
# @Time    : 2021/8/9 下午3:33
# @Author  : ruima
# @File    : dataset.py
# @Software: PyCharm

import collections
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

# Transforms
train_transforms = transforms.Compose([
    transforms.ToTensor()
])
test_transforms = transforms.Compose([
    transforms.ToTensor()
])


def image_score(gt_file):
    """Add image and score to table."""
    table = []
    with open(gt_file) as f:
        lines = f.readlines()

    for line in lines:
        score = float(line.split(' ')[0])
        img_name = line.split(' ')[1][:-1]
        table.append([img_name, score])

    return np.array(table)


def score_cal(gt_file):
    """Find max and min scores."""
    score_all = []
    with open(gt_file) as f:
        lines = f.readlines()

    for line in lines:
        score = float(line.split(' ')[0])
        score_all.append(score)

    score_all = np.array(score_all)
    score_min = np.min(score_all)
    score_max = np.max(score_all)

    return score_min, score_max


class Datasets(data.Dataset):
    """Datasets processing."""
    def __init__(self, root, txtroot, transform=None, train_phase=True, seed=0):
        self.score_min, self.score_max = score_cal(txtroot)
        self.root = root
        self.txtroot = txtroot
        self.files = collections.defaultdict(list)
        gt_table = image_score(self.txtroot)
        gt_table = np.random.RandomState(seed=seed).permutation(gt_table)

        trainnum = int(len(gt_table) * 0.8)
        train_sel = gt_table[0:trainnum]
        test_sel = gt_table[trainnum:]

        if train_phase:
            self.phase = 'train'
            train_sel = np.random.RandomState(seed=seed).permutation(train_sel)
            self.files[self.phase] = train_sel
        else:
            self.phase = 'test'
            test_sel = np.random.RandomState(seed=seed).permutation(test_sel)
            self.files[self.phase] = test_sel
        self.transform = transform

    def __getitem__(self, index):
        img_info = self.files[self.phase][index]

        # Load image
        img = Image.open(self.root + img_info[0])
        if self.transform is not None:
            img = self.transform(img)

        # Generate score
        tscore = float(img_info[1])
        tscore = (tscore - self.score_min) / (self.score_max - self.score_min)
        score = tscore

        return img, score

    def __len__(self):
        length = len(self.files[self.phase])
        return length


def train_loader(root, txtroot, batch_size, num_workers=0, pin_memory=True):
    """Train loader."""
    return data.DataLoader(
        Datasets(root=root, txtroot=txtroot, transform=train_transforms, train_phase=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)


def test_loader(root, txtroot, batch_size, num_workers=0, pin_memory=True):
    """Test loader."""
    return data.DataLoader(
        Datasets(root=root, txtroot=txtroot, transform=test_transforms, train_phase=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)
