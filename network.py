# -*- coding: utf-8 -*-
# @Time    : 2021/8/9 下午3:35
# @Author  : ruima
# @File    : network.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torchvision import models


def weight_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class ModifiedResNext101(nn.Module):
    """ModifiedResNext101."""

    def __init__(self):
        super().__init__()
        resnet = models.resnext101_32x8d(pretrained=True)

        # Create the shared feature generator.
        self.shared = nn.Sequential()
        for name, module in resnet.named_children():
            if name != 'fc':
                self.shared.add_module(name, module)

        self.shared.add_module(name='conv_n1', module=nn.Conv2d(2048, 1024, kernel_size=1))
        self.shared.add_module(name='bn_n1', module=nn.BatchNorm2d(1024))
        self.shared.add_module(name='relu', module=nn.ReLU(inplace=True))
        self.shared.add_module(name='conv_n2', module=nn.Conv2d(1024, 1, kernel_size=1))

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedResNext101, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages do not update.
        for module in self.shared.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = nn.Sigmoid()(x)
        x = x.view(-1)
        return x


def init_dump(arch_name):
    """Dumps pretrained model in required format."""
    model = ModifiedResNext101()
    previous_masks = {}
    for module_idx, module in enumerate(model.shared.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            mask = torch.ByteTensor(module.weight.data.size()).fill_(1)
            if 'cuda' in module.weight.data.type():
                mask = mask.cuda()
            previous_masks[module_idx] = mask
    torch.save({
        'dataset2idx': {'imagenet': 1},
        'previous_masks': previous_masks,
        'model': model,
    }, './imagenet/{}.pt'.format(arch_name))


if __name__ == '__main__':
    arch_name = 'ModifiedResNext101'
    init_dump(arch_name)
