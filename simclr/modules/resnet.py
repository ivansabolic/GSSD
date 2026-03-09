import torchvision

import models


def get_resnet(name, weights=None):
    if weights == 'None':
        weights = None

    resnets = {
        "resnet18_cifar": models.ResNet(18, num_classes=10),
        "resnet18": torchvision.models.resnet18(weights=weights),
        "resnet50": torchvision.models.resnet50(weights=weights),
        "densenet121": torchvision.models.densenet121(weights=weights),
    }

    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]
