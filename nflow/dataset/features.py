import os.path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from datasets import dataset_num_clases


class GaussianNoiseTransform:
    def __init__(self, std):
        self.std = std

    def __call__(self, x):
        x = x + torch.randn_like(x) * self.std
        return x.numpy()


class SelfSupervisedFeaturesDataset(Dataset):
    def __init__(self, features_path, split, normalize_features=True, transform=None, **kwargs):
        loaded = np.load(features_path)

        self.features = torch.Tensor(loaded[f'{split}_X'])
        if normalize_features:
            self.features = (self.features - self.features.mean(0)) / self.features.std(0)

        self.labels = loaded[f'{split}_y']
        self.transform = transform

    def __getitem__(self, index):
        x = self.features[index]
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.features)


class SubsetFeaturesDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.labels = self.dataset.labels[self.indices]

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


class OneClassDataset(SubsetFeaturesDataset):
    def __init__(self, dataset, target_class):
        if type(target_class) == int:
            self.indices = torch.arange(len(dataset.labels))[torch.Tensor(dataset.labels) == target_class]
        elif isinstance(target_class, Iterable):
            self.indices = torch.arange(len(dataset.labels))[torch.isin(torch.Tensor(dataset.labels), torch.Tensor(list(target_class)))]
        super().__init__(dataset, self.indices)


class UnsupervisedFeaturesDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def get_data_id(args):
    from pathlib import Path
    name = Path(args.dataset).name.split('_')
    return '{}'.format(name[0])


def _get_data_shape(args):
    if 'resnet18' in args.selfsup_backbone:
        return args.n_features_latent if args.n_features_latent != -1 else 512
    elif 'resnet50' in args.selfsup_backbone:
        return 1024
    elif args.selfsup_backbone == 'densenet121':
        return 1024
    raise ValueError(f"Unknown backbone {args.selfsup_backbone}")


def _load_features_path(args):
    features_name = f'extracted_features{args.feats}'
    if args.n_features_latent != -1:
        features_name += f"_latent-{args.n_features_latent}"
    if args.projection_dim != 128:
        features_name += f"_proj-{args.projection_dim}"
    features = os.path.join(args.experiment_dir, f'{features_name}.npz')
    print(f"Loading features from {features}")
    return features


def get_data_oneclass(args):
    features = _load_features_path(args)
    data_shape = _get_data_shape(args)

    feature_train_transforms = []
    if args.gaussian_noise_std != 0:
        feature_train_transforms.append(GaussianNoiseTransform(args.gaussian_noise_std))
    feature_train_transforms.append(transforms.Lambda(lambda x: torch.Tensor(x)))
    feature_train_transforms = transforms.Compose(feature_train_transforms)
    feature_test_transforms = transforms.Lambda(lambda x: torch.Tensor(x))

    train_dataset_all = SelfSupervisedFeaturesDataset(features, 'train', transform=feature_train_transforms)
    # Use clean (unfiltered) test set for per-class evaluation so all classes have samples
    clean_test_dataset_all = SelfSupervisedFeaturesDataset(features, 'clean_test', transform=feature_test_transforms)

    all_classes = set(range(dataset_num_clases[args.dataset]))
    train_dataset = UnsupervisedFeaturesDataset(OneClassDataset(train_dataset_all, args.oneclass))
    train_dataset_other = UnsupervisedFeaturesDataset(OneClassDataset(train_dataset_all, all_classes - {args.oneclass}))
    test_dataset = UnsupervisedFeaturesDataset(OneClassDataset(clean_test_dataset_all, args.oneclass))
    test_dataset_other = UnsupervisedFeaturesDataset(OneClassDataset(clean_test_dataset_all, all_classes - {args.oneclass}))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    train_loader_other = DataLoader(train_dataset_other, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader_other = DataLoader(test_dataset_other, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    print(f"Data shape: {data_shape}")
    return [train_loader, train_loader_other], [test_loader, test_loader_other], data_shape


def get_data_allclasses(args, unsupervised=False):
    features = _load_features_path(args)
    data_shape = _get_data_shape(args)

    feature_transforms = transforms.Compose([
        GaussianNoiseTransform(0.1),
        transforms.Lambda(lambda x: torch.Tensor(x)),
    ])

    train_dataset = SelfSupervisedFeaturesDataset(features, 'train', normalize_features=False, transform=feature_transforms)
    test_dataset = SelfSupervisedFeaturesDataset(features, 'test', normalize_features=False, transform=feature_transforms)
    clean_test_dataset = SelfSupervisedFeaturesDataset(features, 'clean_test', normalize_features=False, transform=feature_transforms)

    test_dataset.features = test_dataset.features[clean_test_dataset.labels != args.target_label]
    test_dataset.labels = test_dataset.labels[clean_test_dataset.labels != args.target_label]

    if unsupervised:
        train_dataset = UnsupervisedFeaturesDataset(train_dataset)
        test_dataset = UnsupervisedFeaturesDataset(test_dataset)
        clean_test_dataset = UnsupervisedFeaturesDataset(clean_test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    train_loader_noshuffle = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    clean_test_loader = DataLoader(clean_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    return [train_loader, train_loader_noshuffle], [test_loader, clean_test_loader], data_shape
