from typing import Tuple

import torchvision


class Dataset:
    def __init__(
        self,
        dataset: torchvision.datasets.VisionDataset,
        image_size: Tuple[int],
        num_labels: int,
        num_train_data: int,
        num_test_data: int,
    ):
        self.dataset = dataset
        self.image_size = image_size
        self.num_labels = num_labels
        self.num_train_data = num_train_data
        self.num_test_data = num_test_data


MNIST = Dataset(
    dataset=torchvision.datasets.MNIST,
    image_size=(1, 28, 28),
    num_labels=10,
    num_train_data=60000,
    num_test_data=10000,
)

CIFAR10 = Dataset(
    dataset=torchvision.datasets.CIFAR10,
    image_size=(3, 32, 32),
    num_labels=10,
    num_train_data=50000,
    num_test_data=10000,
)

CIFAR100 = Dataset(
    dataset=torchvision.datasets.CIFAR100,
    image_size=(3, 32, 32),
    num_labels=100,
    num_train_data=50000,
    num_test_data=10000,
)

IMAGENET1K = Dataset(
    dataset=torchvision.datasets.ImageFolder,
    image_size=(3, 224, 224),
    num_labels=1000,
    num_train_data=0,
    num_test_data=50000,
)

choices = {
    "mnist": MNIST,
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "imagenet1k": IMAGENET1K,
}
