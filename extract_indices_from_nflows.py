import pickle
import time
import warnings

import numpy as np
import torch
import torchvision
from torchvision import transforms
import argparse
import os
from tqdm import tqdm

from datasets import dataset_num_clases
from datasets.attacks.utils import OneClassImageFolder
from nflow.dataset.features import SelfSupervisedFeaturesDataset, GaussianNoiseTransform
from nflow.model.mini_flow import get_model


def get_supervised_loader(args):
    """Load the original (clean) supervised dataset to get ground-truth labels."""
    test_transform = transforms.Compose([transforms.ToTensor()])

    if args.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=args.original_dataset_dir, train=True, transform=test_transform, download=False)
    elif args.dataset == "cifar100":
        dataset = torchvision.datasets.CIFAR100(root=args.original_dataset_dir, train=True, transform=test_transform, download=False)
    elif args.dataset == "gtsrb":
        dataset = torchvision.datasets.GTSRB(root=args.original_dataset_dir, split="train", transform=test_transform, download=False)
    elif args.dataset == "tinyimagenet":
        subset = torch.arange(30)
        dataset = OneClassImageFolder(target_class=subset, root=args.original_dataset_dir + "/tiny-imagenet-200/train", transform=test_transform)
    elif args.dataset == "imagenet":
        dataset = torchvision.datasets.ImageFolder(root=args.original_dataset_dir + "/imagenet30/train", transform=test_transform)
    elif args.dataset == "vggface2":
        dataset = torchvision.datasets.ImageFolder(root=args.original_dataset_dir + "/vggface2/train", transform=test_transform)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported")

    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


def load_features(args):
    """Load extracted features from .npz file and create data loaders."""
    features_name = f'extracted_features{args.feats}'
    if args.n_features_latent != -1:
        features_name += f"_latent-{args.n_features_latent}"
    if args.projection_dim != 128:
        features_name += f"_proj-{args.projection_dim}"

    features_path = os.path.join(args.experiment_dir, f'{features_name}.npz')
    print(f"Loading features from {features_path}")

    feat_transforms = [transforms.Lambda(lambda x: torch.Tensor(x))]
    feat_transforms = transforms.Compose(feat_transforms)
    feat_test_transforms = transforms.Lambda(lambda x: torch.Tensor(x))

    train_dataset = SelfSupervisedFeaturesDataset(features_path, 'train', transform=feat_transforms)
    test_dataset = SelfSupervisedFeaturesDataset(features_path, 'test', transform=feat_test_transforms)
    clean_test_dataset = SelfSupervisedFeaturesDataset(features_path, 'clean_test', transform=feat_test_transforms)

    if args.attack != "cbd":
        test_dataset.features = test_dataset.features[clean_test_dataset.labels != args.target_label]
        test_dataset.labels = test_dataset.labels[clean_test_dataset.labels != args.target_label]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    clean_test_loader = torch.utils.data.DataLoader(clean_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if 'resnet18' in args.selfsup_backbone:
        data_shape = args.n_features_latent if args.n_features_latent != -1 else 512
    elif 'resnet50' in args.selfsup_backbone:
        data_shape = 1024
    elif args.selfsup_backbone == 'densenet121':
        data_shape = 1024
    else:
        raise ValueError(f"Unknown backbone {args.selfsup_backbone}")

    return train_loader, test_loader, clean_test_loader, data_shape


def load_flow_models(args, flows_dir, num_classes, data_shape):
    """Load per-class normalizing flow models from checkpoint directory."""
    model_paths = os.listdir(flows_dir)
    models = [None] * num_classes
    for model_path in sorted(model_paths):
        if not model_path.startswith("MF_oneclass"):
            continue
        class_idx = int(model_path.split("-")[-1])
        with open(os.path.join(flows_dir, model_path, 'args.pickle'), 'rb') as f:
            model_args = pickle.load(f)
            model_args.model = args.model
        model = get_model(model_args, data_shape=data_shape)
        checkpoint = torch.load(os.path.join(flows_dir, model_path, 'check', 'checkpoint.pt'), map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        model.to(args.device)
        models[class_idx] = model

    return models


def compute_log_likelihoods(args, models, loader):
    """Compute per-class log-likelihoods for all samples using flow models."""
    with torch.no_grad():
        p_xs = torch.zeros(len(loader.dataset), args.num_classes)
        preds = torch.zeros(len(loader.dataset))
        for i, x in enumerate(tqdm(loader, total=len(loader))):
            p_x = torch.stack(
                [models[j].log_prob(x.to(args.device)) for j in range(args.num_classes)]
            ).cpu().transpose(0, 1)

            p_xs[i * args.batch_size: (i + 1) * args.batch_size] = p_x
            preds[i * args.batch_size: (i + 1) * args.batch_size] = p_x.exp().argmax(1)

    return preds, p_xs


def get_indices_for_label_consistent(adv_dataset_dir):
    """Get sample indices for label-consistent attack datasets."""
    class_paths = os.listdir(adv_dataset_dir)
    indices = []
    for class_path in sorted(class_paths):
        imgs = os.listdir(os.path.join(adv_dataset_dir, class_path))
        for img in sorted(imgs):
            indices.append(int(img.split(".")[0]))
    return np.array(indices)


def get_poisoned_set_indices(args):
    """Load indices of poisoned samples from dataset metadata."""
    if args.attack != "issba":
        return torch.load(f"{args.poisoned_set}.pt")
    else:
        poisoned_set_path = os.path.join(args.poisoned_set + "_issba", "poisoned_set.npy")
        return np.load(poisoned_set_path)


def detect_non_distribution_classes(args, p_xs, poisoned_labels):
    """Detect classes with non-distribution poisoning (out-of-distribution triggers).

    Uses histogram analysis of max likelihood from other-class flows to find
    classes where a subset of samples have anomalously low likelihood under
    non-matching flows.
    """
    poisoned_classes = []
    for class_idx in range(args.num_classes):
        p_xs_class = p_xs[poisoned_labels == class_idx]
        non_class_mask = torch.ones(args.num_classes).bool()
        non_class_mask[class_idx] = False
        p_xs_other = p_xs_class[:, non_class_mask].exp().max(1)[0]

        bins = 30
        hist_counts, hist_edges = np.histogram(p_xs_other, bins=bins)
        below_lambda = np.arange(len(hist_edges))[hist_edges < args.lambda_]
        if len(below_lambda) <= 1:
            continue
        first_below = below_lambda[-1]
        minimum = hist_counts[:first_below].argmin()
        sum_below = hist_counts[:minimum].sum()
        if minimum != 0 and sum_below > args.beta_d * len(p_xs_other) and sum_below >= 2 * hist_counts[minimum]:
            poisoned_classes.append(class_idx)

    return poisoned_classes


def detect_in_distribution_classes(args, p_xs, poisoned_labels, exclude_classes):
    """Detect classes with in-distribution poisoning.

    Uses average likelihood of this-class flow on other-class samples.
    If the average likelihood is too high, the class is likely poisoned.
    """
    poisoned_classes = []
    for class_idx in range(args.num_classes):
        if class_idx in exclude_classes:
            continue

        p_xs_class_flow = p_xs[:, class_idx]
        p_xs_other = p_xs_class_flow[poisoned_labels != class_idx]

        if -p_xs_other.mean() < args.beta_nd:
            poisoned_classes.append(class_idx)
        print(f"Class {class_idx}: NLL this={-p_xs_class_flow[poisoned_labels == class_idx].mean():.3f}, "
              f"NLL other={-p_xs_other.mean():.3f}")

    return poisoned_classes


def compute_score(p_xs, target_class, num_classes):
    """Compute the detection score s(z) = log(p(z|y_T) / max_{y != y_T} p(z|y))."""
    other_mask = torch.ones(num_classes).bool()
    other_mask[target_class] = False
    p_xs_exp = p_xs.exp()
    diff = p_xs_exp[:, target_class] / p_xs_exp[:, other_mask].max(1).values
    return diff.log()


def extract_class_indices(p_xs, poisoned_labels, class_idx, num_classes,
                          clean_threshold, poisoned_threshold, is_in_distribution,
                          poisoned_set=None, clean_set=None, source_labels=None):
    """Extract clean and poisoned indices for a single class.

    Args:
        is_in_distribution: If True, samples with highest score are clean (in-distribution).
                          If False, samples with lowest score are clean (non-distribution).
    """
    class_mask = poisoned_labels == class_idx
    p_xs_class = p_xs[class_mask]
    class_indices = torch.arange(len(p_xs))[class_mask]

    other_mask = torch.ones(num_classes).bool()
    other_mask[class_idx] = False
    diff = p_xs_class[:, class_idx] - p_xs_class[:, other_mask].max(1).values
    sorted_indices = diff.argsort()
    p_xs_sorted = p_xs_class[sorted_indices]
    class_indices_sorted = class_indices[sorted_indices]

    if is_in_distribution:
        n_clean = int((1 - clean_threshold) * len(sorted_indices))
        clean_idx = class_indices_sorted[n_clean:]
        n_poisoned = int(poisoned_threshold * len(sorted_indices))
        poisoned_idx = class_indices_sorted[:n_poisoned]
        poisoned_relabel_slice = slice(None, n_poisoned)
    else:
        n_clean = int(clean_threshold * len(sorted_indices))
        clean_idx = class_indices_sorted[:n_clean]
        n_poisoned = int((1 - poisoned_threshold) * len(sorted_indices))
        poisoned_idx = class_indices_sorted[n_poisoned:]
        poisoned_relabel_slice = slice(n_poisoned, None)

    print(f"  Class {class_idx}: {len(clean_idx)} clean, {len(poisoned_idx)} poisoned")

    # Log detection quality if ground truth is available
    if poisoned_set is not None and class_idx == (poisoned_labels[poisoned_set].mode().values.item() if poisoned_set.any() else -1):
        ps = poisoned_set[class_mask][sorted_indices]
        cs = clean_set[class_mask][sorted_indices]
        if is_in_distribution:
            print(f"  Poisoned in clean set: {ps[n_clean:].sum()}/{len(clean_idx)}")
            print(f"  Clean in poisoned set: {cs[:n_poisoned].sum()}/{len(poisoned_idx)}")
        else:
            print(f"  Poisoned in clean set: {ps[:n_clean].sum()}/{len(clean_idx)}")
            print(f"  Clean in poisoned set: {cs[n_poisoned:].sum()}/{len(poisoned_idx)}")

    # Relabel poisoned samples using likelihood from other flows
    p_xs_sorted_killed = p_xs_sorted.exp().clone()
    p_xs_sorted_killed[:, class_idx] = 0
    relabeled = p_xs_sorted_killed[poisoned_relabel_slice].argmax(1)

    if is_in_distribution:
        return clean_idx.cpu().numpy(), poisoned_idx.cpu().numpy(), relabeled
    else:
        return clean_idx.cpu().numpy(), None, None


def extract_and_save_indices(args, p_xs, poisoned_labels, source_labels,
                             non_dist_classes, in_dist_classes,
                             poisoned_set, clean_set, out_dir):
    """Extract clean/poisoned indices for all detected classes and save to disk."""
    if len(non_dist_classes) == 0 and len(in_dist_classes) == 0:
        print("No poisoned classes detected, nothing to extract.")
        return

    modified = {}
    for cls in in_dist_classes:
        print(f"Extracting in-distribution class {cls}:")
        clean, poisoned, relabeled = extract_class_indices(
            p_xs, poisoned_labels, cls, args.num_classes,
            args.clean_threshold, args.poisoned_threshold,
            is_in_distribution=True,
            poisoned_set=poisoned_set, clean_set=clean_set, source_labels=source_labels)
        modified[cls] = (clean, poisoned, relabeled)

        if poisoned is not None:
            acc = (relabeled == source_labels[torch.from_numpy(poisoned).long()]).float().mean().item()
            print(f"  Relabeling accuracy: {acc:.3f}")

    for cls in non_dist_classes:
        print(f"Extracting non-distribution class {cls}:")
        clean, poisoned, relabeled = extract_class_indices(
            p_xs, poisoned_labels, cls, args.num_classes,
            args.clean_threshold, args.poisoned_threshold,
            is_in_distribution=False,
            poisoned_set=poisoned_set, clean_set=clean_set, source_labels=source_labels)
        modified[cls] = (clean, None, None)

    # Aggregate results
    remove_labels = []
    cleans, cleans_labels = [], []
    poisoneds, poisoneds_labels = [], []
    for label, (clean_idx, poisoned_idx, relabeled) in modified.items():
        remove_labels.append(label)
        cleans.extend(clean_idx)
        cleans_labels.extend(np.ones(len(clean_idx), dtype=np.uint8) * label)
        if poisoned_idx is not None:
            poisoneds.extend(poisoned_idx)
            poisoneds_labels.extend(relabeled.cpu().numpy())

    additional_indices = np.concatenate([cleans, poisoneds]) if poisoneds else np.array(cleans)
    additional_labels = np.concatenate([cleans_labels, poisoneds_labels]) if poisoneds else np.array(cleans_labels)

    indices_name = (f"indices_cthr-{args.clean_threshold}_pthr-{args.poisoned_threshold}"
                    f"_beta_nd-{args.beta_nd}_beta_d-{args.beta_d}_lambda-{args.lambda_}.npz")

    np.savez(
        os.path.join(out_dir, indices_name),
        remove_labels=remove_labels,
        additional_indices=additional_indices,
        additional_labels=additional_labels,
        clean_indices=cleans,
        clean_labels=cleans_labels,
        relabeled_indices=poisoneds,
        relabeled_labels=poisoneds_labels,
    )
    print(f"Saved indices to {os.path.join(out_dir, indices_name)}")


def main(args):
    args.num_classes = dataset_num_clases[args.dataset]
    print(args)

    # Load data
    train_loader, test_loader, clean_test_loader, data_shape = load_features(args)

    flows_name = f"nflows_{args.model}_{args.steps}_{args.inflate_coef}"
    flows_dir = os.path.join(args.experiment_dir, flows_name)
    print(f"Loading flows from {flows_dir}")

    models = load_flow_models(args, flows_dir, args.num_classes, data_shape)
    for model in models:
        model.eval()

    # Compute log-likelihoods
    start_time = time.time()
    sup_loader = get_supervised_loader(args)
    source_labels = torch.zeros(len(sup_loader.dataset))
    for i, (_, y) in enumerate(sup_loader):
        source_labels[i] = y

    preds, p_xs = compute_log_likelihoods(args, models, train_loader)
    preds_test, p_xs_test = compute_log_likelihoods(args, models, test_loader)
    preds_clean_test, p_xs_clean_test = compute_log_likelihoods(args, models, clean_test_loader)
    print(f"Inference time: {time.time() - start_time:.1f}s")

    # Build poisoned/clean label sets
    poisoned_labels = torch.Tensor(train_loader.dataset.labels)
    poisoned_set_indices = get_poisoned_set_indices(args)
    poisoned_set = torch.zeros(len(source_labels)).bool()
    poisoned_set[poisoned_set_indices] = True

    if args.attack == "labelconsistent":
        eps = 16
        lc_indices = get_indices_for_label_consistent(
            os.path.join(args.experiment_dir, f"adv_dataset_eps{eps}", 'target_adv_dataset'))
        source_labels = source_labels[lc_indices]
        poisoned_set = poisoned_set[lc_indices]

    if args.attack == "issba":
        source_labels = torch.cat([source_labels[~poisoned_set], source_labels[poisoned_set]])
        poisoned_set = torch.cat([poisoned_set[~poisoned_set], poisoned_set[poisoned_set]])

    clean_set = ~poisoned_set
    clean_set[poisoned_labels != args.target_label] = False

    print(f"Poisoned samples: {poisoned_set.sum()}")
    print(f"Clean target-class samples: {clean_set.sum()}")

    # Print generative classifier metrics
    print(f"Train ACC (source labels): {(preds == source_labels).float().mean():.4f}")
    print(f"Train ACC (poisoned labels): {(preds == poisoned_labels).float().mean():.4f}")
    print(f"Train ASR: {(preds[poisoned_set] == poisoned_labels[poisoned_set]).float().mean():.4f}")
    print(f"Clean test ACC: {(preds_clean_test == torch.from_numpy(clean_test_loader.dataset.labels)).float().mean():.4f}")
    if args.attack != "cbd":
        print(f"Test ASR: {(preds_test == torch.from_numpy(test_loader.dataset.labels)).float().mean():.4f}")

    # Detect poisoned classes
    print("\n--- Non-distribution poisoning detection ---")
    non_dist_classes = detect_non_distribution_classes(args, p_xs, poisoned_labels)

    print("\n--- In-distribution poisoning detection ---")
    in_dist_classes = detect_in_distribution_classes(args, p_xs, poisoned_labels, non_dist_classes)

    if non_dist_classes:
        print(f"\nNon-distribution poisoned classes: {non_dist_classes}")
    if in_dist_classes:
        print(f"In-distribution poisoned classes: {in_dist_classes}")
    if not non_dist_classes and not in_dist_classes:
        print("\nNo poisoned classes detected!")

    # Extract and save indices
    extract_and_save_indices(
        args, p_xs, poisoned_labels, source_labels,
        non_dist_classes, in_dist_classes,
        poisoned_set, clean_set, flows_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect poisoned samples using normalizing flows")
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--attack', type=str, default='badnets')
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--original_dataset_dir', type=str, required=True)
    parser.add_argument('--target_label', type=int, required=True)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--poisoned_set', type=str, default=None)
    parser.add_argument('--clean_threshold', type=float, default=0.15)
    parser.add_argument('--poisoned_threshold', type=float, default=0.15)
    parser.add_argument('--selfsup_backbone', type=str, default='resnet18')
    parser.add_argument('--model', type=str, default="nflow")
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--inflate_coef', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--feats', type=str, default='')
    parser.add_argument('--beta_nd', type=float, default=0.6)
    parser.add_argument('--beta_d', type=float, default=0.05)
    parser.add_argument('--lambda_', type=float, default=0.75)
    parser.add_argument('--n_features_latent', type=int, default=-1)
    parser.add_argument('--projection_dim', type=int, default=128)

    main(parser.parse_args())
