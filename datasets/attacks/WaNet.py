'''
This is the implement of WaNet [1].

Reference:
[1] WaNet - Imperceptible Warping-based Backdoor Attack. ICLR 2021.
'''

import copy
import warnings
from copy import deepcopy
import random
from pathlib import Path

import numpy as np
import torch
import PIL
from PIL import Image
from torchvision.transforms import functional as F
import torch.nn as nn
from torchvision.transforms import Compose
from torchvision.datasets import CIFAR10, DatasetFolder, GTSRB, CIFAR100

from .utils import OneClassDataset
from .base import Base

class AddTrigger:
    def __init__(self):
        pass

    def add_trigger(self, img, noise=False):
        """Add WaNet trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).
            noise (bool): turn on noise mode, default is False

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        if noise:
            ins = torch.rand(1, self.h, self.h, 2) * self.noise_rescale - 1  # [-1, 1]
            grid = self.grid + ins / self.h
            grid = torch.clamp(self.grid + ins / self.h, -1, 1)
        else:
            grid = self.grid
        poison_img = nn.functional.grid_sample(img.unsqueeze(0), grid, align_corners=True).squeeze()  # CHW
        return poison_img


class AddCIFAR10Trigger(AddTrigger):
    """Add WaNet trigger to CIFAR10 image.

    Args:
        identity_grid (orch.Tensor): the poisoned pattern shape.
        noise_grid (orch.Tensor): the noise pattern.
        noise (bool): turn on noise mode, default is False.
        s (int or float): The strength of the noise grid. Default is 0.5.
        grid_rescale (int or float): Scale :attr:`grid` to avoid pixel values going out of [-1, 1].
            Default is 1.
        noise_rescale (int or float): Scale the random noise from a uniform distribution on the
            interval [0, 1). Default is 2.
    """

    def __init__(self, identity_grid, noise_grid, noise=False, s=0.5, grid_rescale=1, noise_rescale=2):
        super(AddCIFAR10Trigger, self).__init__()

        self.identity_grid = deepcopy(identity_grid)
        self.noise_grid = deepcopy(noise_grid)
        self.h = self.identity_grid.shape[2]
        self.noise = noise
        self.s = s
        self.grid_rescale = grid_rescale
        grid = self.identity_grid + self.s * self.noise_grid / self.h
        self.grid = torch.clamp(grid * self.grid_rescale, -1, 1)
        self.noise_rescale = noise_rescale

    def __call__(self, img):
        # img = F.pil_to_tensor(img)
        img = torch.Tensor(np.array(img).transpose(2, 0, 1))
        img = F.convert_image_dtype(img, torch.float)

        img = self.add_trigger(img, noise=self.noise)

        # spremi sliku after
        img = img.numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 255).round().astype(np.uint8)

        img = Image.fromarray(img)
        # img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img


class AddDatasetFolderTrigger(AddTrigger):
    """Add WaNet trigger to DatasetFolder images.

    Args:
        identity_grid (orch.Tensor): the poisoned pattern shape.
        noise_grid (orch.Tensor): the noise pattern.
        noise (bool): turn on noise mode, default is False.
        s (int or float): The strength of the noise grid. Default is 0.5.
        grid_rescale (int or float): Scale :attr:`grid` to avoid pixel values going out of [-1, 1].
            Default is 1.
        noise_rescale (int or float): Scale the random noise from a uniform distribution on the
            interval [0, 1). Default is 2.
    """

    def __init__(self, identity_grid, noise_grid, noise=False, s=1, grid_rescale=1, noise_rescale=2):
        super(AddDatasetFolderTrigger, self).__init__()

        self.identity_grid = deepcopy(identity_grid)
        self.noise_grid = deepcopy(noise_grid)
        self.h = self.identity_grid.shape[2]
        self.noise = noise
        self.s = s
        self.grid_rescale = grid_rescale
        grid = self.identity_grid + self.s * self.noise_grid / self.h
        self.grid = torch.clamp(grid * self.grid_rescale, -1, 1)
        self.noise_rescale = noise_rescale




    def __call__(self, img):
        """Get the poisoned image.

        Args:
            img (PIL.Image.Image | numpy.ndarray | torch.Tensor): If img is numpy.ndarray or torch.Tensor, the shape should be (H, W, C) or (H, W).
        Returns:
            torch.Tensor: The poisoned image.
        """
        if type(img) == PIL.Image.Image:
            img = F.pil_to_tensor(img)
            img = F.convert_image_dtype(img, torch.float)
            img = self.add_trigger(img, noise=self.noise)
            # 1 x H x W
            if img.size(0) == 1:
                img = img.squeeze().numpy()
                img = Image.fromarray(np.clip(img*255,0,255).round().astype(np.uint8), mode='L')
            # 3 x H x W
            elif img.size(0) == 3:
                img = img.numpy().transpose(1, 2, 0)
                img = Image.fromarray(np.clip(img*255,0,255).round().astype(np.uint8))
            else:
                raise ValueError("Unsupportable image shape.")
            return img
        elif type(img) == np.ndarray:
            # H x W
            if len(img.shape) == 2:
                img = torch.from_numpy(img)
                img = F.convert_image_dtype(img, torch.float)
                img = self.add_trigger(img, noise=self.noise)
                img = img.numpy()

            # H x W x C
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)
                img = F.convert_image_dtype(img, torch.float)
                img = self.add_trigger(img, noise=self.noise)
                img = img.permute(1, 2, 0).numpy()

            return img
        elif type(img) == torch.Tensor:
            # H x W
            if img.dim() == 2:
                img = F.convert_image_dtype(img, torch.float)
                img = self.add_trigger(img, noise=self.noise)
            # H x W x C
            else:
                img = F.convert_image_dtype(img, torch.float)
                img = img.permute(2, 0, 1)
                img = self.add_trigger(img, noise=self.noise)
                img = img.permute(1, 2, 0)
            return img
        else:
            raise TypeError('img should be PIL.Image.Image or numpy.ndarray or torch.Tensor. Got {}'.format(type(img)))




class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target


class PoisonedCIFAR10(CIFAR10):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 identity_grid,
                 noise_grid,
                 noise,
                 poisoned_transform_index,
                 poisoned_target_transform_index,
                 poisoned_set=None,
                 noise_set=None):
        super(PoisonedCIFAR10, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=False)
        total_num = len(benign_dataset)
        if poisoned_set:
            poisoned_set = f"{poisoned_set}.pt"
        if poisoned_set and Path(poisoned_set).exists():
            self.poisoned_set = torch.load(poisoned_set)
        else:
            tmp_list = torch.arange(total_num)
            tmp_list = tmp_list[torch.tensor(self.targets) != y_target]
            poisoned_num = int(len(tmp_list) * poisoned_rate)
            assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
            shuffled = torch.randperm(len(tmp_list))
            tmp_list = tmp_list[shuffled]
            self.poisoned_set = tmp_list[:poisoned_num]
            if poisoned_set:
                torch.save(self.poisoned_set, poisoned_set)

        # add noise
        self.noise = noise
        if noise_set:
            noise_set = f"noise_set_{str(poisoned_rate).replace('.', '-')}.pt"
        if noise_set and Path(noise_set).exists():
            self.noise_set = torch.load(noise_set)
        else:
            noise_rate = poisoned_rate * 2
            noise_num = int(total_num * noise_rate)
            # uzeti razliku izmedu tmp_list i self.poisoned_set
            tmp_list = torch.arange(total_num)
            tmp_list = tmp_list[torch.tensor(self.targets) != y_target]

            if not isinstance(self.poisoned_set, torch.Tensor):
                self.poisoned_set = torch.Tensor(self.poisoned_set)
            combined = torch.cat((self.poisoned_set, tmp_list, tmp_list))
            uniques, counts = combined.unique(return_counts=True)
            tmp_list = uniques[counts == 2]

            shuffled = torch.randperm(len(tmp_list))
            tmp_list = tmp_list[shuffled]
            self.noise_set = tmp_list[:noise_num]
            if noise_set:
                torch.save(self.noise_set, noise_set)

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
            self.poisoned_transform_noise = Compose([])  # add noise
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
            self.poisoned_transform_noise = copy.deepcopy(self.transform)  # add noise
        self.poisoned_transform.transforms.insert(poisoned_transform_index,
                                                  AddCIFAR10Trigger(identity_grid, noise_grid, noise=False))
        # add noise transform
        self.poisoned_transform_noise.transforms.insert(poisoned_transform_index,
                                                        AddCIFAR10Trigger(identity_grid, noise_grid, noise=True))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        # print(index in self.poisoned_set)
        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        # add noise mode
        elif index in self.noise_set and self.noise == True:
            img = self.poisoned_transform_noise(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


class PoisonedCIFAR100(CIFAR100):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 identity_grid,
                 noise_grid,
                 noise,
                 poisoned_transform_index,
                 poisoned_target_transform_index,
                 poisoned_set=None,
                 noise_set=None,
                 **kwargs):
        super(PoisonedCIFAR100, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=False)
        total_num = len(benign_dataset)
        if poisoned_set:
            poisoned_set = f"{poisoned_set}.pt"
        if poisoned_set and Path(poisoned_set).exists():
            self.poisoned_set = torch.load(poisoned_set)
        else:
            tmp_list = torch.arange(total_num)
            tmp_list = tmp_list[torch.tensor(self.targets) != y_target]
            poisoned_num = int(len(tmp_list) * poisoned_rate)
            assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
            shuffled = torch.randperm(len(tmp_list))
            tmp_list = tmp_list[shuffled]
            self.poisoned_set = tmp_list[:poisoned_num]
            if poisoned_set:
                torch.save(self.poisoned_set, poisoned_set)

        # add noise
        self.noise = noise
        if noise_set:
            noise_set = f"noise_set_{str(poisoned_rate).replace('.', '-')}.pt"
        if noise_set and Path(noise_set).exists():
            self.noise_set = torch.load(noise_set)
        else:
            noise_rate = poisoned_rate * 2
            noise_num = int(total_num * noise_rate)
            # uzeti razliku izmedu tmp_list i self.poisoned_set
            tmp_list = torch.arange(total_num)
            tmp_list = tmp_list[torch.tensor(self.targets) != y_target]

            combined = torch.cat((self.poisoned_set, tmp_list, tmp_list))
            uniques, counts = combined.unique(return_counts=True)
            tmp_list = uniques[counts == 2]

            shuffled = torch.randperm(len(tmp_list))
            tmp_list = tmp_list[shuffled]
            self.noise_set = tmp_list[:noise_num]
            if noise_set:
                torch.save(self.noise_set, noise_set)

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
            self.poisoned_transform_noise = Compose([])  # add noise
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
            self.poisoned_transform_noise = copy.deepcopy(self.transform)  # add noise
        self.poisoned_transform.transforms.insert(poisoned_transform_index,
                                                  AddCIFAR10Trigger(identity_grid, noise_grid, noise=False))
        # add noise transform
        self.poisoned_transform_noise.transforms.insert(poisoned_transform_index,
                                                        AddCIFAR10Trigger(identity_grid, noise_grid, noise=True))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        # print(index in self.poisoned_set)
        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        # add noise mode
        elif index in self.noise_set and self.noise == True:
            img = self.poisoned_transform_noise(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


class PoisonedGTSRB(GTSRB):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 identity_grid,
                 noise_grid,
                 noise,
                 poisoned_transform_index,
                 poisoned_target_transform_index,
                 poisoned_set=None,
                 noise_set=None):
        super(PoisonedGTSRB, self).__init__(
            benign_dataset.root,
            benign_dataset._split,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=False)
        self.targets = [label for _, label in self._samples]

        total_num = len(benign_dataset)
        if poisoned_set:
            poisoned_set = f"{poisoned_set}.pt"
        if poisoned_set and Path(poisoned_set).exists():
            self.poisoned_set = torch.load(poisoned_set)
        else:
            tmp_list = torch.arange(total_num)
            tmp_list = tmp_list[torch.tensor(self.targets) != y_target]
            poisoned_num = int(len(tmp_list) * poisoned_rate)
            assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
            shuffled = torch.randperm(len(tmp_list))
            tmp_list = tmp_list[shuffled]
            self.poisoned_set = tmp_list[:poisoned_num]
            if poisoned_set:
                torch.save(self.poisoned_set, poisoned_set)

        # add noise
        self.noise = noise
        if noise_set:
            noise_set = f"noise_set_{str(poisoned_rate).replace('.', '-')}.pt"
        if noise_set and Path(noise_set).exists():
            self.noise_set = torch.load(noise_set)
        else:
            noise_rate = poisoned_rate * 2
            noise_num = int(total_num * noise_rate)
            # uzeti razliku izmedu tmp_list i self.poisoned_set
            tmp_list = torch.arange(total_num)
            tmp_list = tmp_list[torch.tensor(self.targets) != y_target]

            combined = torch.cat((self.poisoned_set, tmp_list, tmp_list))
            uniques, counts = combined.unique(return_counts=True)
            tmp_list = uniques[counts == 2]

            shuffled = torch.randperm(len(tmp_list))
            tmp_list = tmp_list[shuffled]
            self.noise_set = tmp_list[:noise_num]
            if noise_set:
                torch.save(self.noise_set, noise_set)

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
            self.poisoned_transform_noise = Compose([])  # add noise
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
            self.poisoned_transform_noise = copy.deepcopy(self.transform)  # add noise
        self.poisoned_transform.transforms.insert(poisoned_transform_index,
                                                  AddDatasetFolderTrigger(identity_grid, noise_grid, noise=False))
        # add noise transform
        self.poisoned_transform_noise.transforms.insert(poisoned_transform_index,
                                                        AddDatasetFolderTrigger(identity_grid, noise_grid, noise=True))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        path, target = self._samples[index]
        img = PIL.Image.open(path).convert("RGB")

        # print(index in self.poisoned_set)
        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        # add noise mode
        elif index in self.noise_set and self.noise == True:
            img = self.poisoned_transform_noise(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target

class PoisonedDatasetFolder(DatasetFolder):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 identity_grid,
                 noise_grid,
                 noise,
                 poisoned_transform_index,
                 poisoned_target_transform_index,
                 poisoned_set = None,
                 noise_set = None):
        super(PoisonedDatasetFolder, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None)
        self.samples = benign_dataset.samples
        self.targets = [s[1] for s in self.samples]

        total_num = len(benign_dataset)
        if poisoned_set:
            poisoned_set = f"{poisoned_set}.pt"
        if poisoned_set and Path(poisoned_set).exists():
            self.poisoned_set = torch.load(poisoned_set)
        else:
            tmp_list = torch.arange(total_num)
            tmp_list = tmp_list[torch.tensor(self.targets) != y_target]
            poisoned_num = int(len(tmp_list) * poisoned_rate)
            assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
            shuffled = torch.randperm(len(tmp_list))
            tmp_list = tmp_list[shuffled]
            self.poisoned_set = tmp_list[:poisoned_num]
            if poisoned_set:
                torch.save(self.poisoned_set, poisoned_set)

        # add noise
        self.noise = noise
        if noise_set:
            noise_set = f"noise_set_{str(poisoned_rate).replace('.', '-')}.pt"
            warnings.warn(f"noise_set is saved_noised set is deprecates")
        if noise_set and Path(noise_set).exists():
            self.noise_set = torch.load(noise_set)
        else:
            noise_rate = poisoned_rate * 2
            noise_num = int(total_num * noise_rate)
            # uzeti razliku izmedu tmp_list i self.poisoned_set
            tmp_list = torch.arange(total_num)
            tmp_list = tmp_list[torch.tensor(self.targets) != y_target]

            combined = torch.cat((torch.Tensor(self.poisoned_set), tmp_list, tmp_list))
            uniques, counts = combined.unique(return_counts=True)
            tmp_list = uniques[counts == 2]

            shuffled = torch.randperm(len(tmp_list))
            tmp_list = tmp_list[shuffled]
            self.noise_set = tmp_list[:noise_num]
            if noise_set:
                torch.save(self.noise_set, noise_set)

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
            self.poisoned_transform_noise = Compose([])  # add noise
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
            self.poisoned_transform_noise = copy.deepcopy(self.transform)  # add noise
        self.poisoned_transform.transforms.insert(poisoned_transform_index,
                                                  AddDatasetFolderTrigger(identity_grid, noise_grid,
                                                                          noise=False))
        # add noise transform
        self.poisoned_transform_noise.transforms.insert(poisoned_transform_index,
                                                        AddDatasetFolderTrigger(identity_grid, noise_grid,
                                                                                noise=True))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index,
                                                         ModifyTarget(y_target))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if index in self.poisoned_set:
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target


def create_poisoned_dataset(dataset, *args, **kwargs):
    if isinstance(dataset, CIFAR100):
        return PoisonedCIFAR100(dataset, *args, **kwargs)
    elif isinstance(dataset, CIFAR10):
        return PoisonedCIFAR10(dataset, *args, **kwargs)
    elif isinstance(dataset, GTSRB):
        return PoisonedGTSRB(dataset, *args, **kwargs)
    elif isinstance(dataset, DatasetFolder):
        return PoisonedDatasetFolder(dataset, *args, **kwargs)
    else:
        raise NotImplementedError


class WaNet(Base):
    """Construct poisoned datasets with BadNets method.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        y_target (int): N-to-1 attack target label.
        poisoned_rate (float): Ratio of poisoned samples.
        pattern (None | torch.Tensor): Trigger pattern, shape (C, H, W) or (H, W).
        weight (None | torch.Tensor): Trigger pattern weight, shape (C, H, W) or (H, W).
        poisoned_transform_train_index (int): The position index that poisoned transform will be inserted in train dataset. Default: 0.
        poisoned_transform_test_index (int): The position index that poisoned transform will be inserted in test dataset. Default: 0.
        poisoned_target_transform_index (int): The position that poisoned target transform will be inserted. Default: 0.
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 model,
                 loss,
                 y_target,
                 poisoned_rate,
                 identity_grid,
                 noise_grid,
                 noise,
                 weight=None,
                 poisoned_transform_train_index=0,
                 poisoned_transform_test_index=0,
                 poisoned_target_transform_index=0,
                 poisoned_set=None,
                 test_poisoned_set=None,
                 schedule=None,
                 seed=0,
                 deterministic=False):
        super(WaNet, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)

        self.poisoned_train_dataset = create_poisoned_dataset(
            train_dataset,
            y_target,
            poisoned_rate,
            identity_grid,
            noise_grid,
            noise,
            poisoned_transform_train_index,
            poisoned_target_transform_index,
            poisoned_set=poisoned_set,
        )

        self.poisoned_test_dataset = create_poisoned_dataset(
            test_dataset,
            y_target,
            1.0,
            identity_grid,
            noise_grid,
            noise,
            poisoned_transform_test_index,
            poisoned_target_transform_index)

        num_classes = len(set(self.poisoned_train_dataset.targets))
        self.asr_dataset = OneClassDataset(
            self.poisoned_test_dataset,
            set(range(num_classes)) - {y_target},
        )

        if not hasattr(self.test_dataset, 'targets'):
            self.test_dataset.targets = [label for _, label in self.test_dataset._samples]
        self.target_class_dataset = OneClassDataset(
            self.test_dataset,
            y_target,
        )


