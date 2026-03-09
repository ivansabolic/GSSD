import os
import time

import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tqdm import tqdm

from extract_indices_from_nflows import get_indices_for_label_consistent
from simclr import SimCLR
from simclr.modules import get_resnet
from simclr.modules import LARS


def train(args, train_loader, model, criterion, optimizer, scaler):
    loss_epoch = 0
    start_time = time.time()
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        # x_i = x_i.cuda(non_blocking=True)
        x_i = x_i.to(args.device)
        # x_j = x_j.cuda(non_blocking=True)
        x_j = x_j.to(args.device)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            # positive pair, with encoding
            h_i, h_j, z_i, z_j = model(x_i, x_j)
            loss = criterion(z_i, z_j)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}\t Time: {time.time() - start_time}")
            start_time = time.time()

        args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def load_optimizer(args, model):

    scheduler = None
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # TODO: LARS
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "LARS":
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * args.batch_size / 256
        optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0, last_epoch=-1
        )
    else:
        raise NotImplementedError

    return optimizer, scheduler


def save_model(args, model, optimizer):
    checkpoint_folder_name = "checkpoints"
    if args.n_features_latent != -1:
        checkpoint_folder_name += "_latent-{}".format(args.n_features_latent)
    if args.projection_dim != 128:
        checkpoint_folder_name += "_proj-{}".format(args.projection_dim)
    out_dir = os.path.join(
        "experiments",
        args.name,
        checkpoint_folder_name,
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out = os.path.join(
        out_dir, "checkpoint_{}.tar".format(args.current_epoch))

    torch.save(model.state_dict(), out)

    symlink = os.path.join(out_dir, "latest.tar")
    if os.path.exists(symlink):
        os.unlink(symlink)
    os.symlink("checkpoint_{}.tar".format(args.current_epoch), symlink)



def load_datasets(args, transforms):
    if args.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=transforms
        )
        clean_test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transforms
        )
    elif args.dataset == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=True, download=True, transform=transforms
        )
        clean_test_dataset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=transforms
        )
    elif args.dataset == "gtsrb":
        train_dataset = torchvision.datasets.GTSRB(root=args.data_dir, split="train", download=True, transform=transforms)
        clean_test_dataset = torchvision.datasets.GTSRB(root=args.data_dir, split="test", download=True, transform=transforms)

    elif args.dataset == "imagenet":
        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(args.data_dir, "imagenet30", "train"), transform=transforms
        )
        clean_test_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(args.data_dir, "imagenet30", "test"), transform=transforms
        )
    else:
        raise NotImplementedError

    return train_dataset, clean_test_dataset

def create_model(args):
    encoder = get_resnet(args.resnet)
    if "densenet" in args.resnet:
        n_features = encoder.classifier.in_features  # get dimensions of fc layer
    elif "cifar" in args.resnet:
        n_features = encoder.linear.in_features
    else:
        n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder, args.projection_dim, n_features, args.n_features_latent)

    return simclr_model, n_features


def load_model(args, simclr_model):
    checkpoints_folder = "checkpoints"
    if args.n_features_latent != -1:
        checkpoints_folder += "_latent-{}".format(args.n_features_latent)
    if args.projection_dim != 128:
        checkpoints_folder += "_proj-{}".format(args.projection_dim)
    model_fp = os.path.join(args.experiment_dir, checkpoints_folder, "latest.tar")

    print(f"Loading model from {model_fp}")
    if not os.path.exists(model_fp):
        raise ValueError(f"Checkpoint '{model_fp}' does not exist")

    simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device))
    simclr_model.eval()

    return simclr_model


def plot_latent_space(args, simclr_model, loader, epoch):
    with torch.no_grad():
        features = []
        labels = []
        # tmp_dir = "tmp_test"
        # os.makedirs(tmp_dir, exist_ok=True)

        for i, (x, y) in enumerate(
                tqdm(loader, total=len(loader))):
            # for j, x_ in enumerate(x):
            #     torchvision.utils.save_image(x_, os.path.join(tmp_dir, f"img_{j}.png"))
            # breakpoint()
            x = x.to(args.device)
            h, _, _, _ = simclr_model(x, x)
            features.extend(h.cpu().numpy())
            labels.extend(y.cpu().numpy())
        features = np.array(features)
        labels = np.array(labels)

    plt.figure(figsize=(20, 20))

    poisoned_set_indices = torch.load(f"{args.train_poisoned_set}.pt")
    poisoned_set = torch.zeros(len(features)).bool()
    poisoned_set[poisoned_set_indices] = True

    # get indices for label consistent
    if args.attack == "labelconsistent":
        eps = 16
        indices = get_indices_for_label_consistent(
            os.path.join(args.experiment_dir, "adv_dataset_eps{}".format(eps), 'target_adv_dataset'))
        poisoned_set = poisoned_set[indices]

    np.random.seed(42)

    dim_red_alg = 'tsne'
    if dim_red_alg == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=0)
    elif dim_red_alg == 'umap':
        import umap
        reducer = umap.UMAP(random_state=0)
    else:
        raise NotImplementedError

    unique_labels = np.unique(labels)
    colors = matplotlib.colormaps['tab20']
    if len(unique_labels) <= 20:
        colors = [colors(i) for i in range(20)]
    else:
        colors_b = matplotlib.colormaps['tab20b']
        colors_c = matplotlib.colormaps['tab20c']
        colors = [colors_b(i) for i in range(20)] + [colors_c(i) for i in range(20)] + [colors(i) for i in range(20)]

    out_dir= os.path.join(
        "experiments",
        args.name,
        "simclr_latent_spaces",
    )
    os.makedirs(out_dir, exist_ok=True)
    # if args.subsample < 1.:
    #     indices = np.random.choice(len(labels), int(args.subsample * len(labels)), replace=False)
    #     reduced_features = features[indices]
    #     reduced_labels = labels[indices]
    #     reduced_poisoned_set = poisoned_set[indices

    reduced_features = features
    reduced_labels = labels
    reduced_poisoned_set = poisoned_set

    plt.figure(figsize=(20, 20))
    reduced_features = reducer.fit_transform(reduced_features)
    for i in range(len(unique_labels)):
        plt.scatter(reduced_features[reduced_labels == unique_labels[i], 0],
                    reduced_features[reduced_labels == unique_labels[i], 1], color=colors[i],
                    label=unique_labels[i], alpha=0.5)

    plt.scatter(reduced_features[reduced_poisoned_set, 0], reduced_features[reduced_poisoned_set, 1], color='black',
                label='poisoned', alpha=0.35)

    plt.title(f'Selfsup Latent space ')
    plt.legend()
    plt.savefig(f'{out_dir}/selfsup_latent_space_epoch{epoch}.png')
    plt.close()
    print(f"Saved {out_dir}/selfsup_latent_space.png")


class LinearDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_linear_dataset(args, simclr_model, loader):
    features = []
    labels = []

    with torch.no_grad():
        for i, (x, y) in enumerate(
                loader):
            x = x.to(args.device)
            h, _, _, _ = simclr_model(x, x)
            features.extend(h.cpu().numpy())
            labels.extend(y.cpu().numpy())
        features = np.array(features)
        labels = np.array(labels)

    linear_dataset = LinearDataset(features, labels)
    return linear_dataset


def test_linear_probe(args, simclr_model, train_loader, test_loader, n_features):
    linear_cfg = {
        "epochs": 100,
        "lr": 0.1,
        "weight_decay": 0,
        "lr_decay_steps": [60, 80],
    }

    # add a linear classifier on top of the encoder
    model = torch.nn.Linear(n_features, args.num_classes)
    model = model.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=linear_cfg["lr"], weight_decay=linear_cfg["weight_decay"])

    features_train_dataset = create_linear_dataset(args, simclr_model, train_loader)
    features_test_dataset = create_linear_dataset(args, simclr_model, test_loader)

    train_loader = torch.utils.data.DataLoader(features_train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(features_test_dataset, batch_size=args.batch_size, shuffle=False)

    for epoch in range(linear_cfg["epochs"]):
        loss_epoch = 0
        for step, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(args.device)
            y = y.to(args.device)

            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

        print(f"Epoch [{epoch}/{linear_cfg['epochs']}]\t Loss: {loss_epoch / len(train_loader)}", end="\r")

    def test(loader):
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for x, y in loader:
                x = x.to(args.device)
                y = y.to(args.device)

                output = model(x)

                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        acc = correct / total
        return acc

    train_acc = test(train_loader)
    test_acc = test(test_loader)

    print(f"Train accuracy: {train_acc}, Test accuracy: {test_acc}")