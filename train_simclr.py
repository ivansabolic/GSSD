import argparse
import os
import time

import torch
import numpy as np
import torchvision

import wandb

import datasets
from datasets.images_dataset import get_poisoned_datasets
from retraining import get_clean_dataset
from utils import yaml_config_hook
from datasets import get_poisoned_images_dataset


from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.train_utils import load_optimizer, train, save_model, plot_latent_space, test_linear_probe


def main(args):
    args.name = "_".join(
        [
            args.dataset,
            args.attack,
            str(args.poisoned_rate),
            str(args.target_label),
            str(args.resnet),
        ]
    )
    print(args)
    args.num_classes = datasets.dataset_num_clases[args.dataset]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    transforms = TransformsSimCLR(size=args.image_size)
    if args.resnet == "resnet18" and args.dataset == "gtsrb":
        transforms.insert(0, torchvision.transforms.Resize(size=(48, 48)))
        transforms.test_transform.transforms.insert(0, torchvision.transforms.Resize(size=(48, 48)))
    # load dataset
    train_loader, test_loader = get_poisoned_images_dataset(
        args.dataset,
        args.data_dir,
        args.attack,
        args.target_label,
        args.poisoned_rate,
        args,
        transforms=transforms,
        experiment_dir=os.path.join("experiments", args.name),
    )

    train_dataset_test, _ = get_poisoned_datasets(
        args.dataset,
        args.data_dir,
        args.attack,
        args.target_label,
        args.poisoned_rate,
        args,
        transforms=transforms.test_transform,
        experiment_dir=os.path.join("experiments", args.name),
    )
    train_loader_noshuffle = torch.utils.data.DataLoader(
        train_dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    _, test_loader_clean = get_clean_dataset(
        args.dataset,
        args.data_dir,
        # args,
        # transforms=transforms.test_transform,
    )

    # initialize ResNet
    encoder = get_resnet(args.resnet, weights=args.weights)
    if args.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
    if "densenet" in args.resnet:
        n_features = encoder.classifier.in_features  # get dimensions of fc layer
    elif "cifar" in args.resnet:
        n_features = encoder.linear.in_features
    else:
        n_features = encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    model = SimCLR(encoder, args.projection_dim, n_features, args.n_features_latent)
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature, 1)

    scaler = torch.cuda.amp.GradScaler()

    if args.wandb:
        out_dir = os.path.join("experiments", args.name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        wandb.init(
            project="simclr",
            name=f"{args.name}_latent-{args.n_features_latent}_proj-{args.projection_dim}",
            id=f"{args.name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
            config=args,
            dir=out_dir,
        )

    start_time = time.time()

    if args.plot_epoch <= args.epochs:
        plot_latent_space(args, model, train_loader_noshuffle, -1)
    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer, scaler)

        if scheduler:
            scheduler.step()

        if args.save_epoch != -1 and epoch % args.save_epoch == 0:
            save_model(args, model, optimizer)

        print(
            f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
        )
        if args.wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "loss": loss_epoch / len(train_loader),
                    "lr": lr,
                }
            )
        args.current_epoch += 1

        if (epoch + 1) % args.plot_epoch == 0:
            plot_latent_space(args, model, train_loader_noshuffle, epoch)

        if (epoch + 1) % args.test_epoch == 0:
            test_linear_probe(args, model, train_loader_noshuffle, test_loader_clean, n_features)

    print(f"Training time: {round(time.time() - start_time, 2)}s")
    ## end training
    save_model(args, model, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Selfsup training module')
    config = yaml_config_hook("./config/simlr_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    main(parser.parse_args())

