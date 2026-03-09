import torchvision
import random
from PIL import ImageFilter


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR.

    Borrowed from https://github.com/facebookresearch/moco/blob/master/moco/loader.py.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))

        return x

class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                # torchvision.transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.4914, 0.4822, 0.4465],
                    [0.2023, 0.1994, 0.2010],
                ),
                # torchvision.transforms.Normalize(
                #     [0.3337, 0.3064, 0.3171],
                #     [0.2672, 0.2564, 0.2629]
                # ),
            ]
        )
        self.transforms = self.train_transform.transforms

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.4914, 0.4822, 0.4465],
                    [0.2023, 0.1994, 0.2010],
                )
                # torchvision.transforms.Normalize(
                #     [0.3337, 0.3064, 0.3171],
                #     [0.2672, 0.2564, 0.2629]
                # ),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

    def insert(self, index, transform):
        self.train_transform.transforms.insert(index, transform)
