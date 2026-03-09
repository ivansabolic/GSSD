import torch
from typing import Iterable

import torchvision


class OneClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, target_class, remove_indices=None, remove_labels=None, additional_indices=None, additional_indices_with_labels=None):
        self.dataset = dataset

        if type(target_class) == int:
            self.target_indices = torch.arange(len(dataset.targets))[torch.Tensor(dataset.targets) == target_class]
        elif isinstance(target_class, Iterable):
            self.target_indices = torch.arange(len(dataset.targets))[torch.isin(torch.Tensor(dataset.targets), torch.Tensor(list(target_class)))]

        # todo brisati poisoned set bez eksplicitnog zadavanja
        # npr. targets = [target for target in self.dataset.targets]
        if remove_indices is not None:
            # set intersection of remove_indices and torch.arange(len(dataset.targets))[torch.Tensor(dataset.targets) == 2]

            # todo this for all2all poisoning
            # remove_indices_unique = torch.Tensor(remove_indices).unique()
            # class_2_unique = torch.arange(len(dataset.targets))[torch.Tensor(dataset.targets) == 1].unique()
            # remove_indices = torch.Tensor([ind for ind in remove_indices_unique if ind in class_2_unique]).long()

            combined = torch.cat([self.target_indices, remove_indices, remove_indices])
            uniques, counts = combined.unique(return_counts=True)
            self.target_indices = uniques[counts == 1].sort()[0]

        if remove_labels is not None:
            filtered_targets = torch.Tensor(self.dataset.targets)[self.target_indices]
            self.target_indices = self.target_indices[~torch.isin(filtered_targets, torch.Tensor(remove_labels))]

        if additional_indices is not None:
            self.target_indices = torch.cat([self.target_indices, additional_indices]).sort()[0]

        self.additional_indices_with_labels_dict = None
        if additional_indices_with_labels is not None:
            self.additional_indices_with_labels_dict = {}
            indices, labels = additional_indices_with_labels
            # Filter out any indices that are in remove_indices to prevent including
            # false-negative poisoned samples detected as clean
            if remove_indices is not None:
                mask = ~torch.isin(indices, remove_indices)
                indices = indices[mask]
                labels = labels[mask]
            self.target_indices = torch.cat([self.target_indices, indices]).sort()[0]

            additional_indices_with_labels_dict = {ind.item(): lab.item() for ind, lab in zip(indices, labels)}
            self.additional_indices_with_labels_dict.update(additional_indices_with_labels_dict)
        
    def __getitem__(self, index):
        index = self.target_indices[index].item()
        x, y = self.dataset[index]

        if self.additional_indices_with_labels_dict is not None:
            if index in self.additional_indices_with_labels_dict:
                y = self.additional_indices_with_labels_dict[index]

        return x, y

    def __len__(self):
        return len(self.target_indices)


class OneClassImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, target_class, **kwargs):
        super(OneClassImageFolder, self).__init__(**kwargs)

        if type(target_class) == int:
            self.target_indices = torch.arange(len(self.targets))[torch.Tensor(self.targets) == target_class]
        elif isinstance(target_class, Iterable):
            self.target_indices = torch.arange(len(self.targets))[
                torch.isin(torch.Tensor(self.targets), torch.Tensor(list(target_class)))]

        self.targets = [self.targets[i] for i in self.target_indices]
        self.samples = [self.samples[i] for i in self.target_indices]
    # def __getitem__(self, index):
    #     index = self.target_indices[index].item()
    #     x, y = self.dataset[index]
    #
    #     return x, y
    #
    # def __len__(self):
    #     return len(self.target_indices)

#
# class StampDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, indices, stamp_ratio, image_size=32):
#         assert isinstance(dataset, OneClassDataset)
#         self.dataset = dataset
#
#
#         # choose stamp_ratio of indices
#         num_stamp_indices = int(len(indices) * stamp_ratio)
#         stamp_indices = torch.randperm(len(indices))[:num_stamp_indices]
#         self.stamp_indices = indices[stamp_indices]
#
#         from .BadNets import AddCIFAR10Trigger
#         self.trigger = torch.zeros((3, image_size, image_size))
#         self.trigger[:, :2, -2:] = 255
#         self.weight = torch.zeros((3, image_size, image_size))
#         self.weight[:, :2, -2:] = 1
#         self.trigger_transform = AddCIFAR10Trigger(self.trigger, self.weight)
#
#
#
#     def __getitem__(self, index):
#         x, y = self.dataset[index]
#         index = self.
#         if index in self.stamp_indices:
#             x = self.trigger_transform(x)



class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        index = self.indices[index].item()
        x, y = self.dataset[index]

        return x, y

    def __len__(self):
        return len(self.indices)

