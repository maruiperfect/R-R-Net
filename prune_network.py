# -*- coding: utf-8 -*-
# @Time    : 2021/1/8 上午10:29
# @Author  : marui
# @File    : prune_network.py

import torch
import argparse
import torch.nn as nn
from prune import SparsePruner
from network import ModifiedResNext101


class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, model, previous_masks):
        # Parameters
        self.args = args
        self.model = model
        self.previous_masks = previous_masks

        # Prune Function
        self.pruner = SparsePruner(
            self.model, self.args.prune_perc_per_layer, self.previous_masks, self.args.train_biases, self.args.train_bn,
            self.args.first_or_not)

    def save_model(self, savename):
        """Saves model to file."""
        base_model = self.model
        ckpt = {
            'previous_masks': self.pruner.current_masks,
            'model': base_model,
        }
        # save to file
        torch.save(ckpt, savename)

    def prune(self):
        """Perform pruning."""
        self.pruner.first_prune()
        self.check(True)
        self.save_model(self.args.save_prefix)

    def check(self, verbose=False):
        """Makes sure that the layers are pruned."""
        print('Checking...')
        for layer_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_zero = weight.view(-1).eq(0).sum()
                if verbose:
                    print('Layer #%d: Pruned %d/%d (%.2f%%)' %
                          (layer_idx, num_zero, num_params, 100 * num_zero / num_params))


def main(args):
    ckpt = torch.load(args.loadname)
    model = ckpt['model']
    previous_masks = ckpt['previous_masks']

    if args.device == torch.device('cuda'):
        model.cuda()

    manager = Manager(args, model, previous_masks)
    manager.prune()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='CL-IQA')
    parser.add_argument('--device', type=str, default=device,
                        help='Device')
    parser.add_argument('--mode', type=str,
                        help='Run mode')
    parser.add_argument('--dataset', type=str,
                        help='Name of dataset')
    parser.add_argument('--loadname', type=str,
                        help='Location to load model')
    parser.add_argument('--lr', type=float,
                        help='Learning rate')
    parser.add_argument('--lr_decay_every', type=int,
                        help='Period of learning rate change')
    parser.add_argument('--lr_decay_factor', type=float,
                        help='Multiplier for learning rate change')
    parser.add_argument('--finetune_epochs', type=int,
                        help='Number of initial finetuning epochs')
    parser.add_argument('--save_prefix', type=str,
                        help='Location to save model')
    parser.add_argument('--prune_perc_per_layer', type=float, default=0.75,
                        help='% of neurons to prune per layer')
    parser.add_argument('--post_prune_epochs', type=int, default=0,
                        help='Number of epochs to finetune for after pruning')
    parser.add_argument('--train_data_loader',
                        help='train_data_loader')
    parser.add_argument('--test_data_loader',
                        help='test_data_loader')
    parser.add_argument('--train_biases', type=bool, default=False,
                        help='use separate biases or not')
    parser.add_argument('--train_bn', type=bool, default=False,
                        help='train batch norm or not')
    parser.add_argument('--disable_pruning_mask', type=bool, default=False,
                        help='use masking or not')
    parser.add_argument('--dataset_idx', type=int,
                        help='dataset_idx')
    parser.add_argument('--first_or_not', type=bool,
                        help='first_or_not')
    parser.add_argument('--similarity', type=list, default=[],
                        help='similarity')
    args = parser.parse_args()

    # Important parameters
    options = {
        'prune_perc': 0.90,
        'loadname': '/home/ruima/work/ACM/Code/github_v2/imagenet/ModifiedResNext101.pt',
        'savename': '/home/ruima/work/ACM/Code/github_v2/imagenet/pre_pruned_ModifiedResNext101.pt',
    }

    # first_prune
    loadname = options['loadname']
    savename = options['savename']

    args.first_or_not = True
    args.mode = 'first_prune'
    args.loadname = loadname
    args.save_prefix = savename
    args.prune_perc_per_layer = options['prune_perc']
    main(args)
