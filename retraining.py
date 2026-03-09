import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, Normalize, RandomCrop, \
    RandomResizedCrop, Resize, CenterCrop, RandomRotation

import models
from datasets.attacks.utils import OneClassDataset
from datasets.attacks import BadNets, Blended, WaNet
from datasets.images_dataset import get_poisoned_dataset_kwargs, get_data_shape
from datasets import dataset_num_clases
from utils import yaml_config_hook


def get_clean_dataset(dataset, dataset_dir):
    if dataset == 'cifar10':
        transform_train = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True,
                                               transform=transform_test)
    elif dataset == 'cifar100':
        transform_train = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            RandomRotation(15),
            ToTensor(),
            Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = Compose([
            ToTensor(),
            Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        trainset = torchvision.datasets.CIFAR100(root=dataset_dir, train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=dataset_dir, train=False, download=True,
                                                  transform=transform_test)

    elif 'imagenet' in dataset:
        transform_train = Compose([
                RandomResizedCrop(224),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        transform_test = Compose([
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        if dataset == 'imagenet':
            dataset_name = 'imagenet30'
        elif dataset == 'imagenet-100':
            dataset_name = 'imagenet-100'
        elif dataset == 'imagenet1k':
            dataset_name = 'imagenet1k'
        else:
            raise NotImplementedError

        trainset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, dataset_name, 'train'),
                                                    transform=transform_train)
        testset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, dataset_name, 'test'),
                                                   transform=transform_test)

    elif dataset == 'vggface2':
        transform_train = Compose([
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'vggface2', 'train'),
                                                    transform=transform_train)
        testset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'vggface2', 'test'),
                                                    transform=transform_test)

    elif dataset == 'gtsrb':
        transform_train = Compose([
            Resize((48, 48)),
            ToTensor(),
        ])
        transform_test = Compose([
            Resize((48, 48)),
            ToTensor(),
        ])

        trainset = torchvision.datasets.GTSRB(root=dataset_dir, split='train', download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.GTSRB(root=dataset_dir, split='test', download=True,
                                                  transform=transform_test)
    else:
        raise NotImplementedError

    return trainset, testset


def get_poisoned_test_transform_index(dataset):
    if dataset == 'cifar10' or dataset == 'cifar100':
        return 0
    elif dataset == 'gtsrb':
        return 1
    elif 'imagenet' in dataset or dataset == 'vggface2':
        return 2
    else:
        raise NotImplementedError


def main(args):
    print(args)
    trainset, testset = get_clean_dataset(args.dataset, args.dataset_dir)

    data_shape = get_data_shape(args.dataset)
    poisoning_params = get_poisoned_dataset_kwargs(args.attack, args.target_label, args.poisoned_rate, data_shape,
                                                   dataset=args.dataset, experiment_dir=args.experiment_dir, device=args.device)
    num_classes = dataset_num_clases[args.dataset]

    if args.dataset == 'imagenet':
        model = torchvision.models.resnet18(weights=None, num_classes=30)
    elif args.dataset == 'imagenet-100':
        model = torchvision.models.resnet18(weights=None, num_classes=100)
    elif args.dataset == 'imagenet1k':
        model = torchvision.models.resnet18(weights=None, num_classes=1000)
    elif args.dataset == 'vggface2':
        model = torchvision.models.densenet121(weights=None, num_classes=30)
    elif args.dataset == 'cifar100':
        model = models.ResNet18C100()
    elif args.dataset == 'cifar10' or args.dataset == 'gtsrb':
        model = models.ResNet(18, num_classes=num_classes)

    experiment_params = {
        "train_dataset": trainset,
        "test_dataset": testset,
        "model": model,
        "loss": nn.CrossEntropyLoss(),
        "poisoned_transform_train_index": poisoning_params.pop("poisoned_transform_index"),
        **poisoning_params,
        "poisoned_transform_test_index": get_poisoned_test_transform_index(args.dataset),
        "poisoned_set": args.poisoned_set,
        "test_poisoned_set": args.test_poisoned_set,
        "schedule": None,
        "seed": 42,
    }

    if args.attack == 'badnets':
        experiment = BadNets(**experiment_params)
    elif args.attack == 'blended':
        experiment = Blended(**experiment_params)
    elif args.attack == 'wanet':
        experiment = WaNet(**experiment_params)
    else:
        raise NotImplementedError

    if args.defend or args.finetune_with_relabeled:
        poisoned_set = torch.load(f"{args.poisoned_set}.pt")
        print(f"Loading indices from {args.indices}")
        indices = np.load(args.indices)
        remove_labels = indices["remove_labels"]
        additional_indices = torch.Tensor(indices["additional_indices"]).long()
        additional_labels = torch.Tensor(indices["additional_labels"]).long()

        if args.clean_only:
            print("Adding only clean indices")
            additional_indices = torch.Tensor(indices["clean_indices"]).long()
            additional_labels = torch.Tensor(indices["clean_labels"]).long()

        if args.finetune_with_relabeled:
            print("Finetuning with relabeled")
            remove_labels = np.arange(num_classes)
            additional_indices = torch.Tensor(indices["relabeled_indices"]).long()
            additional_labels = torch.Tensor(indices["relabeled_labels"]).long()
            if len(additional_indices) == 0:
                print("No relabeled samples found, skipping fine-tuning step.")
                return

        if args.attack == "issba":
            experiment.purify(remove_labels, (additional_indices, additional_labels))
        else:
            if isinstance(poisoned_set, np.ndarray):
                poisoned_set = torch.Tensor(poisoned_set).long()

            experiment.poisoned_train_dataset = OneClassDataset(
                experiment.poisoned_train_dataset,
                set(range(dataset_num_clases[args.dataset])),
                remove_indices=poisoned_set,
                remove_labels=remove_labels,
                additional_indices_with_labels=(additional_indices, additional_labels),
            )

    lr = 0.1
    weight_decay = 5e-4
    if args.finetune_with_relabeled:
        milestones = [10, 20]
        lr = 1e-4
    elif 'imagenet' in args.dataset or args.dataset == 'vggface2':
        args.epochs = 90
        milestones = [60]
        lr = 0.01
        weight_decay = 1e-4
    else:
        milestones = [100, 150]

    add_exp_info = 'clean' if (args.clean_only or args.finetune_with_relabeled) and args.defend else ''

    schedule = {
        'device': args.device,
        'benign_training': False,
        'defense': args.defend,
        'indices': args.indices,
        'batch_size': 32 if args.dataset == 'vggface2' else 128,
        'num_workers': 4,

        'lr': lr,
        'momentum': 0.9,
        'weight_decay': weight_decay,
        'gamma': 0.1,
        'schedule': milestones,

        'epochs': args.epochs,
        'resume': args.finetune_with_relabeled,

        'log_iteration_interval': 100,
        'wandb': args.wandb,
        'test_epoch_interval': 1 if args.finetune_with_relabeled else 5,
        'save_epoch_interval': 200,

        'experiment_dir': args.experiment_dir,
        'add_exp_info': add_exp_info,
    }

    start = time.time()
    experiment.train(schedule)
    print(f"Training took {time.time() - start} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/retraining_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument("--defend", action="store_true")
    parser.add_argument("--clean_only", action="store_true")
    parser.add_argument("--finetune_with_relabeled", action="store_true")
    parser.add_argument("--work_dir", required=True)

    main(parser.parse_args())
