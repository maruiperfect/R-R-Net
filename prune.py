# -*- coding: utf-8 -*-
# @Time    : 2021/8/9 下午3:34
# @Author  : ruima
# @File    : prune.py
# @Software: PyCharm

import torch
import torch.nn as nn


class SparsePruner(object):
    """Performs pruning on the given model."""

    def __init__(self, model, prune_perc, previous_masks, train_bias, train_bn, first):
        self.model = model
        self.prune_perc = prune_perc
        self.train_bias = train_bias
        self.train_bn = train_bn

        self.current_masks = None
        self.previous_masks = previous_masks
        valid_key = list(previous_masks.keys())[0]
        self.current_dataset_idx = previous_masks[valid_key].max()
        self.first = first

    def pruning_mask(self, weights, previous_mask, layer_idx):
        """Ranks weights by magnitude. Sets all below kth to 0.
           Returns pruned mask.
        """
        self.current_dataset_idx = self.current_dataset_idx.cuda()
        # Select all prunable weights, ie. belonging to current dataset.
        previous_mask = previous_mask.cuda()
        tensor = weights[previous_mask.eq(self.current_dataset_idx)]
        abs_tensor = tensor.abs()
        cutoff_rank = round(self.prune_perc * tensor.numel())
        # cutoff_value = abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0][0]
        if cutoff_rank == 0:
            cutoff_rank = 1

        cutoff_value = abs_tensor.view(-1).cpu()
        cutoff_value = cutoff_value.kthvalue(cutoff_rank)[0]
        cutoff_value = cutoff_value.cuda()

        # Remove those weights which are below cutoff and belong to current
        # dataset that we are training for.
        remove_mask = weights.abs().le(cutoff_value) * previous_mask.eq(self.current_dataset_idx)

        previous_mask[remove_mask.eq(1)] = 0
        mask = previous_mask
        print('Layer #%d, pruned %d/%d (%.2f%%) (Total in layer: %d)' %
              (layer_idx, mask.eq(0).sum(), tensor.numel(),
               100 * mask.eq(0).sum() / tensor.numel(), weights.numel()))
        return mask

    def first_prune(self):
        """Prune before training task1."""
        print('Pruning for dataset idx: %d' % self.current_dataset_idx)
        assert not self.current_masks, 'Current mask is not empty? Pruning twice?'
        self.current_masks = {}

        print('Pruning each layer by removing %.2f%% of values' %
              (100 * self.prune_perc))
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.pruning_mask(
                    module.weight.data, self.previous_masks[module_idx], module_idx)
                self.current_masks[module_idx] = mask.cuda()

    def prune(self):
        """Gets pruning mask for each layer, based on previous_masks.
           Sets the self.current_masks to the computed pruning masks.
        """
        print('Pruning for dataset idx: %d' % self.current_dataset_idx)
        assert not self.current_masks, 'Current mask is not empty? Pruning twice?'
        self.current_masks = {}

        print('Pruning each layer by removing %.2f%% of values' %
              (100 * self.prune_perc))
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.pruning_mask(
                    module.weight.data, self.previous_masks[module_idx], module_idx)
                self.current_masks[module_idx] = mask.cuda()
                # Set pruned weights to 0.
                weight = module.weight.data
                weight[self.current_masks[module_idx].eq(0)] = 0.0

    def pruning_previous_mask(self, weights, previous_mask, layer_idx, prune_ratio, dataset_idx):
        """Ranks weights by magnitude. Sets all below kth to 0.
           Returns pruned mask.
        """
        dataset_idx = torch.tensor(dataset_idx, dtype=torch.uint8)
        dataset_idx = dataset_idx.cuda()
        # Select all prunable weights, ie. belonging to current dataset.
        previous_mask = previous_mask.cuda()
        tensor = weights[previous_mask.eq(dataset_idx)]
        abs_tensor = tensor.abs()
        cutoff_rank = int(round(prune_ratio * tensor.numel()))

        if cutoff_rank == 0:
            cutoff_rank = 1

        cutoff_value = abs_tensor.view(-1).cpu()
        cutoff_value = cutoff_value.kthvalue(cutoff_rank)[0]
        cutoff_value = cutoff_value.cuda()

        # Remove those weights which are below cutoff and belong to current
        # dataset that we are training for.
        remove_mask = weights.abs().le(cutoff_value) * \
                      previous_mask.eq(dataset_idx)

        previous_mask[remove_mask.eq(1)] = 0
        mask = previous_mask
        return mask

    def pre_prune(self, all_prune_ratio):
        """Before training the new task, some parameters belonging to old tasks are pruned
        according to the relevance between the tasks.
        """
        assert self.current_masks

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[module_idx]
                for idx, prune_ratio in enumerate(all_prune_ratio):
                    if prune_ratio != 0:
                        idx = torch.tensor(idx, dtype=torch.uint8)
                        idx = idx.cuda()
                        tensor = module.weight.data[layer_mask.eq(idx + 2)]
                        if tensor.numel():
                            layer_mask = self.pruning_previous_mask(module.weight.data, layer_mask, module_idx,
                                                                    prune_ratio, idx + 2)

    def test_prune(self, all_prune_ratio):
        assert self.previous_masks

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.previous_masks[module_idx]
                for idx, prune_ratio in enumerate(all_prune_ratio):
                    if prune_ratio != 0:
                        idx = torch.tensor(idx, dtype=torch.uint8)
                        idx = idx.cuda()
                        tensor = module.weight.data[layer_mask.eq(idx + 2)]
                        if tensor.numel():
                            layer_mask = self.pruning_previous_mask(module.weight.data, layer_mask, module_idx,
                                                                    prune_ratio, idx + 2)

    def make_grads_zero(self):
        """Sets grads of fixed weights to 0."""
        assert self.current_masks

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[module_idx]

                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    module.weight.grad.data[layer_mask.ne(
                        self.current_dataset_idx)] = 0
                    if not self.train_bias:
                        # Biases are fixed.
                        if module.bias is not None:
                            module.bias.grad.data.fill_(0)
            elif 'BatchNorm' in str(type(module)):
                # Set grads of batchnorm params to 0.
                if not self.train_bn:
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)

    def make_pruned_zero(self):
        """Makes pruned weights 0."""
        assert self.current_masks

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[module_idx]
                module.weight.data[layer_mask.eq(0)] = 0.0

    def apply_mask(self, dataset_idx):
        """To be done to retrieve weights just for a particular task."""
        dataset_idx = dataset_idx.cuda()
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                mask = self.current_masks[module_idx].cuda()
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(dataset_idx)] = 0.0

    def apply_mask_test(self, dataset_idx):
        """To be done to retrieve weights just for a particular task."""
        dataset_idx = dataset_idx.cuda()
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                mask = self.previous_masks[module_idx].cuda()
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(dataset_idx)] = 0.0

    def restore_biases(self, biases):
        """Use the given biases to replace existing biases."""
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if module.bias is not None:
                    module.bias.data.copy_(biases[module_idx])

    def get_biases(self):
        """Gets a copy of the current biases."""
        biases = {}
        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if module.bias is not None:
                    biases[module_idx] = module.bias.data.clone()
        return biases

    def make_finetuning_mask(self):
        """Turns previously pruned weights into trainable weights for
           current task.
        """
        assert self.previous_masks
        # if not self.first:
        self.current_dataset_idx += 1

        for module_idx, module in enumerate(self.model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.previous_masks[module_idx]
                mask[mask.eq(0)] = self.current_dataset_idx

        self.current_masks = self.previous_masks

    def complete_model_finetune(self, dataset_idx, pre_model, pre_masks, current_model, current_masks):
        dataset_idx = dataset_idx.cuda()
        pre_model_list = list(pre_model.shared.modules())
        for module_idx, module in enumerate(current_model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                pre_weight = pre_model_list[module_idx].weight.data
                mask = current_masks[module_idx].cuda()
                pre_mask = pre_masks[module_idx].cuda()
                weight[mask.ne(dataset_idx)] = pre_weight[pre_mask.ne(0)]

    def complete_model_prune(self, dataset_idx, pre_model, pre_masks, current_model, current_masks):
        dataset_idx = dataset_idx.cuda()
        pre_model_list = list(pre_model.shared.modules())
        for module_idx, module in enumerate(current_model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                pre_weight = pre_model_list[module_idx].weight.data
                mask = current_masks[module_idx].cuda()
                pre_mask = pre_masks[module_idx].cuda()
                weight[pre_mask.ne(dataset_idx)] = pre_weight[pre_mask.ne(dataset_idx)]

    def complete_mask_finetune(self, dataset_idx, pre_masks, current_model, current_masks):
        dataset_idx = dataset_idx.cuda()
        for module_idx, module in enumerate(current_model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = current_masks[module_idx].cuda()
                pre_mask = pre_masks[module_idx].cuda()
                mask[mask.ne(dataset_idx)] = pre_mask[pre_mask.ne(0)]

    def complete_mask_prune(self, dataset_idx, pre_masks, current_model, current_masks):
        dataset_idx = dataset_idx.cuda()
        for module_idx, module in enumerate(current_model.shared.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = current_masks[module_idx].cuda()
                pre_mask = pre_masks[module_idx].cuda()
                mask[pre_mask.ne(dataset_idx)] = pre_mask[pre_mask.ne(dataset_idx)]
