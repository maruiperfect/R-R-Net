# -*- coding: utf-8 -*-
# @Time    : 2021/8/9 下午3:29
# @Author  : ruima
# @File    : main.py
# @Software: PyCharm

import os
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from dataset import train_loader, test_loader
from prune import SparsePruner
from torch.utils.tensorboard import SummaryWriter
from network import ModifiedResNext101


class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, model, previous_masks):
        # Parameters
        self.args = args
        self.model = model
        self.previous_masks = previous_masks

        # Train data loader and test data loader
        self.train_data_loader = self.args.train_data_loader
        self.test_data_loader = self.args.test_data_loader

        # Prune Function
        self.pruner = SparsePruner(
            self.model, self.args.prune_perc_per_layer, self.previous_masks, self.args.train_biases, self.args.train_bn,
            self.args.first)

        # Loss Function
        self.criterion = nn.L1Loss()

        # Optimizer
        self.params_to_optimize = self.model.parameters()
        self.optimizer = optim.SGD(params=self.params_to_optimize, lr=self.args.lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=self.args.lr_decay_every, gamma=self.args.lr_decay_factor)

    def train(self, epochs, train_mode):
        """Performs training."""
        print('-' * 30)
        print('Training Dataset: {}'.format(self.args.dataset))
        print('Training Mode: {}'.format(train_mode))
        print('-' * 30)
        print('Epoch\tTrain loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')

        best_srcc = -1

        if self.args.train_bn:
            self.model.train()
        else:
            self.model.train_nobn()

        for epoch_id in range(epochs):
            epoch_loss, train_srcc = self.do_epoch(
                self.optimizer, self.train_data_loader, self.model, self.criterion)
            writer.add_scalar(self.args.dataset + '_' + train_mode + '_loss', sum(epoch_loss) / len(epoch_loss),
                              epoch_id)
            self.scheduler.step(epoch_id)
            test_srcc, test_plcc = self.eval(self.pruner.current_dataset_idx, self.test_data_loader, self.model)
            writer.add_scalar(self.args.dataset + '_' + train_mode + '_srcc', test_srcc, epoch_id)
            print('%d\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (epoch_id + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))
            if test_srcc > best_srcc:
                print('Best model so far, SRCC: %0.4f -> %0.4f' % (best_srcc, test_srcc))
                best_srcc = test_srcc
                self.save_model(self.args.save_prefix)
            if epoch_id == epochs - 1:
                self.save_complete_model(self.args.save_prefix, self.args.save_prefix + '_complete')

    def do_epoch(self, optimizer, train_data_loader, model, criterion):
        epoch_loss = []
        pscores = []
        tscores = []
        for batch, label in train_data_loader:
            batch = batch.cuda()
            label = label.cuda()

            model.zero_grad()

            output = model(batch)
            loss = criterion(output, label)
            batch_loss = loss.item()
            epoch_loss.append(batch_loss)
            loss.backward()

            pscores = pscores + output.cpu().tolist()
            tscores = tscores + label.cpu().tolist()

            # Set fixed param grads to 0 (Set the label in the mask to not be the gradient at the label position of the
            # current data set to 0)
            if not self.args.disable_pruning_mask:
                self.pruner.make_grads_zero()

            optimizer.step()

            # Set the pruned weight to 0 (set the weight at the position labeled 0 in the mask to 0)
            if not self.args.disable_pruning_mask:
                self.pruner.make_pruned_zero()

        pscores = np.array(pscores)
        pscores = pscores.flatten()
        tscores = np.array(tscores)

        train_srcc, _ = stats.spearmanr(pscores, tscores)

        return epoch_loss, train_srcc

    @torch.no_grad()
    def eval(self, dataset_idx, test_data_loader, model):
        pscores = []
        tscores = []

        # Set the weight of the label equal to 0 in the mask to 0, and set the weight of the label in the mask to be
        # greater than the label of the current data set to 0
        if not self.args.disable_pruning_mask:
            if self.args.mode == 'test':
                self.pruner.apply_mask_test(dataset_idx)
            elif self.args.mode == 'ratio_cal':
                self.pruner.apply_mask_test(dataset_idx)
            else:
                self.pruner.apply_mask(dataset_idx)

        model.eval()
        for batch, label in test_data_loader:
            batch = batch.cuda()
            output = model(batch)
            pscores = pscores + output.cpu().tolist()
            tscores = tscores + label.cpu().tolist()

        pscores = np.array(pscores)
        pscores = pscores.flatten()
        tscores = np.array(tscores)

        test_srcc, _ = stats.spearmanr(pscores, tscores)
        test_plcc, _ = stats.pearsonr(pscores, tscores)

        if self.args.train_bn:
            self.model.train()
        else:
            self.model.train_nobn()

        return test_srcc, test_plcc

    def save_complete_model(self, loadname, savename):
        """Saves complete model to file."""
        pre_ckpt = torch.load(self.args.loadname)
        pre_mask = pre_ckpt['previous_masks']
        pre_model = pre_ckpt['model']

        current_ckpt = torch.load(loadname + '.pt')
        current_mask = current_ckpt['previous_masks']
        current_model = current_ckpt['model']

        if self.args.mode == 'finetune':
            self.pruner.complete_mask_finetune(self.pruner.current_dataset_idx, pre_mask, current_model, current_mask)
            self.pruner.complete_model_finetune(self.pruner.current_dataset_idx, pre_model, pre_mask, current_model,
                                                current_mask)
        elif self.args.mode == 'prune':
            self.pruner.complete_mask_prune(self.pruner.current_dataset_idx, pre_mask, current_model, current_mask)
            self.pruner.complete_model_prune(self.pruner.current_dataset_idx, pre_model, pre_mask, current_model,
                                             current_mask)

        ckpt = {
            'previous_masks': current_mask,
            'model': current_model,
        }

        # save to file
        torch.save(ckpt, savename + '.pt')

    def save_model(self, savename):
        """Saves model to file."""

        ckpt = {
            'previous_masks': self.pruner.current_masks,
            'model': self.model,
        }

        # save to file
        torch.save(ckpt, savename + '.pt')

    def prune(self):
        """Perform pruning."""
        print('-' * 30)
        print('Begin Prune:')
        print('-' * 30)
        self.pruner.prune()
        self.pruner.pre_prune(self.args.ratio)

        test_srcc, test_plcc = self.eval(self.pruner.current_dataset_idx, self.test_data_loader, self.model)

        print('-' * 30)
        print('Test after prune results: srcc = {} , plcc = {}'.format(test_srcc, test_plcc))
        print('-' * 30)

        # Do final finetuning to improve results on pruned network.
        if self.args.post_prune_epochs:
            self.train(epochs=self.args.post_prune_epochs, train_mode='prune')

        print('-' * 30)
        print('Pruning summary:')
        print('-' * 30)
        self.check(True)

    def check(self, verbose=False):
        """Makes sure that the layers are pruned."""
        for layer_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_zero = weight.view(-1).eq(0).sum()
                if verbose:
                    print('Layer #%d: Pruned %d/%d (%.2f%%)' %
                          (layer_idx, num_zero, num_params, 100 * num_zero / num_params))

    def test(self):
        """Testing on previous and current dataset."""
        dataset_idx = self.args.dataset_idx
        dataset_idx = torch.tensor(dataset_idx, dtype=torch.uint8)
        self.pruner.test_prune(self.args.test_ratio)
        test_srcc, test_plcc = self.eval(dataset_idx, self.test_data_loader, self.model)
        return test_srcc, test_plcc

    def ratio_cal(self):
        """Pruning ratio calculation"""
        dataset_idx = self.args.dataset_idx
        dataset_idx = torch.tensor(dataset_idx, dtype=torch.uint8)
        self.pruner.test_prune(self.args.test_ratio)
        test_srcc, test_plcc = self.eval(dataset_idx, self.test_data_loader, self.model)
        ratio = 1 - 1 / (1 + 0.3 * math.e ** (-5 * test_srcc))
        return ratio, test_srcc


def main(args):
    ckpt = torch.load(args.loadname)
    model = ckpt['model']
    previous_masks = ckpt['previous_masks']
    model.cuda()
    manager = Manager(args, model, previous_masks)

    if args.mode == 'ratio_cal':
        ratio, srcc = manager.ratio_cal()
        return ratio, srcc
    elif args.mode == 'finetune':
        manager.pruner.make_finetuning_mask()
        manager.pruner.pre_prune(args.ratio)
        # finetune
        manager.train(epochs=args.finetune_epochs, train_mode='finetune')
    elif args.mode == 'prune':
        # prune
        manager.prune()
    elif args.mode == 'test':
        # test
        test_srcc, test_plcc = manager.test()
        return test_srcc, test_plcc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='R-R-Net')
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
    parser.add_argument('--post_prune_epochs', type=int, default=0,
                        help='Number of epochs to finetune for after pruning')
    parser.add_argument('--save_prefix', type=str,
                        help='Location to save model')
    parser.add_argument('--prune_perc_per_layer', type=float, default=0.75,
                        help='% of neurons to prune per layer')
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
    parser.add_argument('--first', type=bool,
                        help='first')
    parser.add_argument('--ratio', type=list, default=[],
                        help='pruning ratio')
    parser.add_argument('--all_ratio', type=list, default=[],
                        help='all pruning ratio')
    parser.add_argument('--test_ratio', type=list, default=[],
                        help='test pruning ratio')
    parser.add_argument('--srcc', type=list, default=[],
                        help='test srcc')
    parser.add_argument('--all_srcc', type=list, default=[],
                        help='all test srcc')
    args = parser.parse_args()

    # Important parameters
    options = {
        'dataset': 'Group1',
        'batch_size': 20,
        'finetune_epochs': 100,
        'post_prune_epochs': 50,
        'prune_perc': [0.67, 0.5],
        'img_root': '',
        'lr': 0.001,
        'lr_decay_every': 10,
        'lr_decay_factor': 0.8,
        'prev_pruned_savename': None,
        'loadname': './imagenet/ModifiedResNext101.pt',
    }

    # Tensorboard logs
    writer = SummaryWriter(
        log_dir='./runs/' + options['dataset'] + '_epoch{},{}_batchsize{}_lr{}'.format(options['finetune_epochs'],
                                                                                       options['post_prune_epochs'],
                                                                                       options['batch_size'],
                                                                                       options['lr']) + '/')

    # Distortion subsets
    Datasets = []
    # Sequence 1
    if options['dataset'] == 'Group1':
        Datasets = ['LIVE', 'LIVEMD_Part1', 'IVIPC_DQA']
    # Sequence 2
    elif options['dataset'] == 'Group2':
        Datasets = ['Challenge', 'LIVEMD_Part2', 'LIVE']
    # Sequence 3
    elif options['dataset'] == 'Group3':
        Datasets = ['TID2013', 'Challenge', 'SHRQ_Regular']

    # Image path and score path
    options['img_root'] = './data/crop_data/'

    img_root_all = []
    score_root_all = []
    for Dataset in Datasets:
        img_root = options['img_root'] + Dataset + '/'
        img_root_all.append(img_root)
        score_root = img_root + Dataset + '.txt'
        score_root_all.append(score_root)

    # Train loader and test loader
    train_loader_all = []
    test_loader_all = []
    for m in range(len(Datasets)):
        train_loader_all.append(
            train_loader(root=img_root_all[m], txtroot=score_root_all[m], batch_size=options['batch_size']))
        test_loader_all.append(
            test_loader(root=img_root_all[m], txtroot=score_root_all[m], batch_size=options['batch_size']))

    # Train, Prune, Test
    for j in range(len(Datasets)):
        if j == 0:
            args.first = True
        else:
            args.first = False

        # Model save path
        savename = './checkpoints/' + options['dataset'] + '_epoch{},{}_batchsize{}_lr{}'.format(
            options['finetune_epochs'], options['post_prune_epochs'], options['batch_size'], options['lr']) + '/' + \
                   Datasets[j] + '/'
        if not os.path.exists(savename):
            os.makedirs(savename)

        # Results save path
        logname = './logs/' + options['dataset'] + '_epoch{},{}_batchsize{}_lr{}'.format(
            options['finetune_epochs'], options['post_prune_epochs'], options['batch_size'], options['lr']) + '/' + \
                  Datasets[j] + '/'
        if not os.path.exists(logname):
            os.makedirs(logname)

        # Ratio save path
        rationame = './ratio/' + options['dataset'] + '_epoch{},{}_batchsize{}_lr{}'.format(
            options['finetune_epochs'], options['post_prune_epochs'], options['batch_size'], options['lr']) + '/' + \
                    Datasets[j] + '/'
        if not os.path.exists(rationame):
            os.makedirs(rationame)

        # SRCC save path
        srccname = './srcc/' + options['dataset'] + '_epoch{},{}_batchsize{}_lr{}'.format(
            options['finetune_epochs'], options['post_prune_epochs'], options['batch_size'], options['lr']) + '/' + \
                   Datasets[j] + '/'
        if not os.path.exists(srccname):
            os.makedirs(srccname)

        # Fine-tuned model save name
        ft_savename = savename + 'train'

        # Pruned model save name
        pruned_savename = savename + 'prune'

        if j == 0:
            loadname = options['loadname']  # initial model
        else:
            loadname = options['prev_pruned_savename'] + '_complete' + '.pt'  # pruned model

        # Public parameters
        args.dataset = Datasets[j]
        args.train_data_loader = train_loader_all[j]
        args.test_data_loader = test_loader_all[j]

        # Pruning ratio calculation
        if not args.first:
            args.mode = 'ratio_cal'
            args.loadname = loadname
            all_ratio = []
            all_srcc = []
            for p in range(2, j + 2):
                args.dataset_idx = p
                args.test_ratio = args.all_ratio[p - 2]
                ratio, srcc = main(args)
                all_ratio.append(ratio)
                all_srcc.append(srcc)
            args.ratio = all_ratio
            args.srcc = all_srcc
        args.all_ratio.append(args.ratio)
        args.all_srcc.append(args.srcc)
        with open(rationame + Datasets[j] + '.txt', 'a') as file_object:
            file_object.write(Datasets[j] + '\n')
            file_object.write('Pruning Ratio:  ' + str(args.all_ratio) + '\n')
        with open(srccname + Datasets[j] + '.txt', 'a') as file_object:
            file_object.write(Datasets[j] + '\n')
            file_object.write('Test SRCC:  ' + str(args.all_srcc) + '\n')

        # Finetune
        args.mode = 'finetune'
        args.loadname = loadname
        args.lr = options['lr']
        args.lr_decay_factor = options['lr_decay_factor']
        if j < len(options['prune_perc']):
            args.finetune_epochs = options['finetune_epochs']
            args.lr_decay_every = options['lr_decay_every']
        else:
            args.finetune_epochs = options['finetune_epochs'] + options['post_prune_epochs']
            args.lr_decay_every = options['lr_decay_every'] * 2
        args.save_prefix = ft_savename
        main(args)

        # Prune
        if j < len(options['prune_perc']):
            args.mode = 'prune'
            args.loadname = ft_savename + '_complete' + '.pt'
            args.prune_perc_per_layer = options['prune_perc'][j]
            args.post_prune_epochs = options['post_prune_epochs']
            args.save_prefix = pruned_savename
            main(args)
            options['prev_pruned_savename'] = pruned_savename

        # Test
        args.mode = 'test'
        if j < len(options['prune_perc']):
            args.loadname = options['prev_pruned_savename'] + '_complete' + '.pt'
        else:
            args.loadname = ft_savename + '_complete' + '.pt'

        for n in range(j + 1):
            args.dataset_idx = n + 2
            args.test_data_loader = test_loader_all[n]
            args.test_ratio = args.all_ratio[n]
            test_srcc, test_plcc = main(args)
            with open(logname + Datasets[j] + '.txt', 'a') as file_object:
                file_object.write(Datasets[n] + '\n')
                file_object.write('SRCC:  ' + str(test_srcc) + '\n')
                file_object.write('PLCC:  ' + str(test_plcc) + '\n')
