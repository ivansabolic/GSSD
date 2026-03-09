import os
import argparse
import torch
import torchvision
import numpy as np

from simclr.modules.transformations import TransformsSimCLR
from simclr.train_utils import load_datasets, create_model, load_model

from utils import yaml_config_hook
from datasets.images_dataset import get_poisoned_datasets


def inference(loader, simclr_model, device):
    feature_vector = []
    projection_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        projection_vector.extend(z.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, projection_vector, labels_vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/simlr_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    print(args)

    transforms = TransformsSimCLR(size=args.image_size).test_transform

    if args.resnet == "resnet18" and args.dataset == "gtsrb":
        transforms.transforms.insert(0, torchvision.transforms.Resize(size=(48, 48)))

    print(transforms)
    train_dataset, clean_test_dataset = load_datasets(args, transforms)

    train_dataset, test_dataset = get_poisoned_datasets(
        args.dataset,
        args.data_dir,
        args.attack,
        args.target_label,
        args.poisoned_rate,
        args,
        transforms=transforms,
        test_transforms=transforms,
        experiment_dir=args.experiment_dir,
    )
    if args.attack == 'cbd':
        test_dataset = clean_test_dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    clean_test_loader = torch.utils.data.DataLoader(
        clean_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    simclr_model, n_features = create_model(args)
    simclr_model = load_model(args, simclr_model)
    simclr_model = simclr_model.to(args.device)

    print("### Creating features from pre-trained context model ###")
    train_X, train_proj_X, train_y = inference(train_loader, simclr_model, args.device)
    test_X, test_proj_X, test_y = inference(test_loader, simclr_model, args.device)
    clean_test_X, clean_test_proj_X, clean_test_y = inference(clean_test_loader, simclr_model, args.device)

    out_file = "extracted_features"
    if args.n_features_latent != -1:
        out_file += f"_latent-{args.n_features_latent}"
    if args.projection_dim != 128:
        out_file += f"_proj-{args.projection_dim}"
    out_file += ".npz"

    np.savez(
        os.path.join(args.experiment_dir, out_file),
        train_X=train_X,
        train_proj_X=train_proj_X,
        train_y=train_y,
        test_X=test_X,
        test_proj_X=test_proj_X,
        test_y=test_y,
        clean_test_X=clean_test_X,
        clean_test_proj_X=clean_test_proj_X,
        clean_test_y=clean_test_y,
    )

    print("Finished!")
