import os.path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

from .attacks import (
    BadNets, Blended, WaNet,
    BadNetsPoisonedCIFAR10, BadNetsPoisonedCIFAR100, BadNetsPoisonedGTSRB, BadNetsPoisonedDatasetFolder,
    BlendedPoisonedCIFAR10, BlendedPoisonedCIFAR100, BlendedPoisonedGTSRB, BlendedPoisonedDatasetFolder,
    WaNetPoisonedCIFAR10, WaNetPoisonedCIFAR100, WaNetPoisonedGTSRB, WaNetPoisonedDatasetFolder,
)
from .attacks.utils import OneClassImageFolder


def get_poisoned_transform_index(dataset):
    if dataset == "cifar10" or dataset == "cifar100" or dataset == "tinyimagenet":
        return 0
    elif dataset == "gtsrb" or dataset == "imagenet" or dataset == "imagenet-100" or dataset == "vggface2" or dataset == "imagenet1k":
        return 1
    else:
        raise NotImplementedError


def get_pattern_and_weight_badnets(dataset, data_shape):
    _, h, w = data_shape
    if dataset == "cifar10":
        pattern = Image.open("./resources/cifar_1.png").convert("RGB").resize((w, h))
        pattern = np.transpose(np.array(pattern), (2, 0, 1))
        pattern = torch.from_numpy(pattern)
        weight = torch.zeros((1, h, w), dtype=torch.float32)
        weight[:, 4:6, 4:6] = 1.0
    elif dataset == "cifar100":
        pattern = torch.zeros((1, h, w), dtype=torch.uint8)
        pattern[:, -1, -1] = 255
        pattern[:, -1, -2] = 0
        pattern[:, -1, -3] = 255
        pattern[:, -2, -1] = 0
        pattern[:, -2, -2] = 255
        pattern[:, -2, -3] = 0
        pattern[:, -3, -1] = 255
        pattern[:, -3, -2] = 0
        pattern[:, -3, -3] = 255
        weight = torch.zeros((1, h, w), dtype=torch.float32)
        weight[0, -3:, -3:] = 1.0
    elif dataset == "gtsrb":
        pattern = torch.zeros((1, h, w), dtype=torch.uint8)
        pattern[0, -3:, -3:] = 255
        weight = torch.zeros((1, h, w), dtype=torch.float32)
        weight[0, -3:, -3:] = 1.0
    elif "imagenet" in dataset or dataset == "vggface2":
        pattern_size = 32
        pattern_path = "./resources/apple-logo.jpg"
        pattern = np.array(Image.open(pattern_path).convert("RGB").resize((pattern_size, pattern_size)))
        pattern = np.transpose(pattern, (2, 0, 1))
        whole_pattern = np.zeros((3, h, w), dtype=np.uint8)
        whole_pattern[:, :pattern_size, :pattern_size] = pattern
        pattern = whole_pattern
        weight = np.zeros((3, h, w), dtype=np.float32)
        weight[np.nonzero(pattern)] = 1.0
        pattern = torch.from_numpy(pattern)
        weight = torch.from_numpy(weight)
    else:
        raise NotImplementedError

    return pattern, weight


def get_pattern_and_weight_blended(dataset, data_shape):
    _, h, w = data_shape
    if dataset in ("cifar10", "cifar100", "gtsrb"):
        pattern = Image.open("./resources/hello_kitty.png").convert("RGB").resize((w, h))
        pattern = np.transpose(np.array(pattern), (2, 0, 1))
    elif "imagenet" in dataset or dataset == "vggface2":
        pattern = (torch.rand(data_shape) * 255).type(torch.uint8)
    else:
        raise NotImplementedError

    if isinstance(pattern, np.ndarray):
        pattern = torch.from_numpy(pattern)

    weight = torch.ones(data_shape) * 0.1
    return pattern, weight


def get_poisoned_dataset_kwargs(attack, target_label, poisoned_rate, data_shape, **kwargs):
    print("Attacking with {}".format(attack))

    poisoned_transform_index = kwargs.get("poisoned_transform_index", get_poisoned_transform_index(kwargs["dataset"]))
    poisoned_test_transform_index = kwargs.get("poisoned_transform_test_index", poisoned_transform_index)
    poisoned_dataset_kwargs = {
        "y_target": target_label,
        "poisoned_transform_index": poisoned_transform_index,
        "poisoned_target_transform_index": 0,
        "poisoned_transform_test_index": poisoned_test_transform_index,
        "poisoned_rate": poisoned_rate,
    }

    if attack == "badnets":
        pattern, weight = get_pattern_and_weight_badnets(kwargs["dataset"], data_shape)
        poisoned_dataset_kwargs.update({"pattern": pattern, "weight": weight})
    elif attack == "blended":
        pattern, weight = get_pattern_and_weight_blended(kwargs["dataset"], data_shape)
        poisoned_dataset_kwargs.update({"pattern": pattern, "weight": weight})
    elif attack == "wanet":
        if kwargs["dataset"] in ("cifar10", "cifar100", "gtsrb"):
            k = 4
        elif kwargs["dataset"] in ("imagenet", "vggface2", "imagenet1k"):
            k = 224
        else:
            raise NotImplementedError

        height = data_shape[1]
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = F.upsample(ins, size=height, mode="bicubic", align_corners=True)
        noise_grid = noise_grid.permute(0, 2, 3, 1)
        array1d = torch.linspace(-1, 1, steps=height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]
        poisoned_dataset_kwargs.update({
            "noise_grid": noise_grid,
            "identity_grid": identity_grid,
            "noise": False,
        })
    else:
        raise NotImplementedError(f"Attack {attack} not implemented")

    return poisoned_dataset_kwargs


def get_data_shape(dataset):
    if 'cifar' in dataset:
        return (3, 32, 32)
    elif dataset == 'gtsrb':
        return (3, 48, 48)
    elif dataset == 'tinyimagenet':
        return (3, 64, 64)
    elif dataset in ('imagenet', 'imagenet-100', 'imagenet1k', 'vggface2'):
        return (3, 224, 224)
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")


def get_poisoned_datasets(dataset, data_dir, attack, target_label, poisoned_rate, args, transforms=None, test_transforms=None, **kwargs):
    poisoned_dataset_kwargs = get_poisoned_dataset_kwargs(attack, target_label, poisoned_rate, get_data_shape(dataset),
                                                          dataset=dataset, transforms=transforms, **kwargs)

    if dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transforms)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transforms)
        poisoned_dataset = {'badnets': BadNetsPoisonedCIFAR10, 'blended': BlendedPoisonedCIFAR10, 'wanet': WaNetPoisonedCIFAR10}[attack]

    elif dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transforms)
        test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transforms)
        poisoned_dataset = {'badnets': BadNetsPoisonedCIFAR100, 'blended': BlendedPoisonedCIFAR100, 'wanet': WaNetPoisonedCIFAR100}[attack]

    elif dataset == 'gtsrb':
        train_dataset = torchvision.datasets.GTSRB(root=data_dir, split="train", download=True, transform=transforms)
        test_dataset = torchvision.datasets.GTSRB(root=data_dir, split="test", download=True, transform=test_transforms)
        poisoned_dataset = {'badnets': BadNetsPoisonedGTSRB, 'blended': BlendedPoisonedGTSRB, 'wanet': WaNetPoisonedGTSRB}[attack]

    elif dataset == 'tinyimagenet':
        subset = torch.arange(30)
        train_dataset = OneClassImageFolder(target_class=subset, root=data_dir + "/tiny-imagenet-200/train", transform=transforms)
        test_dataset = OneClassImageFolder(target_class=subset, root=data_dir + "/tiny-imagenet-200/val", transform=test_transforms)
        poisoned_dataset = {'badnets': BadNetsPoisonedDatasetFolder}[attack]

    elif dataset in ('imagenet', 'imagenet-100', 'imagenet1k'):
        folder_name = {'imagenet': 'imagenet30', 'imagenet-100': 'imagenet-100', 'imagenet1k': 'imagenet1k'}[dataset]
        train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, folder_name, 'train'), transform=transforms)
        test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, folder_name, 'test'), transform=test_transforms)
        if dataset == 'imagenet':
            poisoned_dataset_kwargs["poisoned_transform_index"] = 1
        poisoned_dataset = {'badnets': BadNetsPoisonedDatasetFolder, 'blended': BlendedPoisonedDatasetFolder, 'wanet': WaNetPoisonedDatasetFolder}[attack]

    elif dataset == 'vggface2':
        train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'vggface2', 'train'), transform=transforms)
        test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'vggface2', 'test'), transform=test_transforms)
        poisoned_dataset_kwargs["poisoned_transform_index"] = 1
        poisoned_dataset = {'badnets': BadNetsPoisonedDatasetFolder, 'blended': BlendedPoisonedDatasetFolder, 'wanet': WaNetPoisonedDatasetFolder}[attack]

    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")

    if args.train_poisoned_set == "":
        raise ValueError("train_poisoned_set must be set")
    train_poisoned_set = args.train_poisoned_set
    test_poisoned_set = args.test_poisoned_set if args.test_poisoned_set != "" else None

    train_kwargs = dict(poisoned_dataset_kwargs)
    train_kwargs.pop("poisoned_transform_test_index")
    train_dataset = poisoned_dataset(train_dataset, poisoned_set=train_poisoned_set, **train_kwargs)

    poisoned_dataset_kwargs["poisoned_rate"] = 1.0
    poisoned_dataset_kwargs["poisoned_transform_index"] = poisoned_dataset_kwargs.pop("poisoned_transform_test_index", 0)
    test_dataset = poisoned_dataset(test_dataset, **poisoned_dataset_kwargs)

    return train_dataset, test_dataset


def get_poisoned_images_dataset(dataset, data_dir, attack, target_label, poisoned_rate, args, transforms=None, test_transforms=None, **kwargs):
    train_dataset, test_dataset = get_poisoned_datasets(dataset, data_dir, attack, target_label, poisoned_rate, args,
                                                        transforms, test_transforms, **kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                               pin_memory=args.pin_memory, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                              pin_memory=args.pin_memory, drop_last=True)

    return train_loader, test_loader
