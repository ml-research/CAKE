import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
import torch
from sklearn import datasets
from torch.utils.data import Dataset, random_split, DataLoader, ConcatDataset
from torchvision.transforms import v2


from torchvision.datasets import FashionMNIST, MNIST, CIFAR10, CelebA, SVHN, CIFAR100

from src.vis import plot_data_and_decision_boundary


@dataclass
class Shape:
    """Represents the shape of an image or tensor.

    Attributes:
        channels (int): The number of channels in the image or tensor.
        height (int): The height of the image or tensor in pixels.
        width (int): The width of the image or tensor in pixels.
    """

    channels: int  # Number of channels
    height: int  # Height in pixels
    width: int  # Width in pixels

    def __iter__(self):
        for element in [self.channels, self.height, self.width]:
            yield element

    def __getitem__(self, index: int):
        return [self.channels, self.height, self.width][index]

    def downscale(self, scale):
        """Downscale this shape by the given scale. Only changes height/width."""
        return Shape(self.channels, round(self.height / scale), round(self.width / scale))

    def upscale(self, scale):
        """Upscale this shape by the given scale. Only changes height/width."""
        return Shape(self.channels, round(self.height * scale), round(self.width * scale))

    @property
    def num_pixels(self):
        return self.width * self.height


def is_grayscale_dataset(dataset_name: str) -> bool:
    return get_data_shape(dataset_name).channels == 1


def is_rgb_dataset(dataset_name: str) -> bool:
    return not is_grayscale_dataset(dataset_name)


def get_number_of_classes(dataset_name: str) -> int:
    """
    Get the number of classes for the given dataset.

    Args:
        dataset_name (str): Dataset name.

    Returns:
        int: Number of classes.
    """
    if "9-clusters" in dataset_name:
        return 9

    if "3-clusters" in dataset_name:
        return 3

    if "synth" in dataset_name:
        return 2

    return {
        "mnist": 10,
        "mnist-32": 10,
        "fmnist": 10,
        "fmnist-32": 10,
        "cifar-2": 2,
        "cifar-10": 10,
        "cifar-100": 100,
        "svhn": 10,
    }[dataset_name]


def get_data_shape(dataset_name: str) -> Shape:
    """Get the expected data shape.

    Args:
        dataset_name (str): Dataset name.

    Returns:
        Tuple[int, int, int]: Tuple of [channels, height, width].
    """
    if "synth" in dataset_name:
        return Shape(1, 2, 1)

    return Shape(
        *{
            "mnist": (1, 28, 28),
            "mnist-32": (1, 32, 32),
            "fmnist": (1, 28, 28),
            "fmnist-32": (1, 32, 32),
            "cifar-2": (3, 32, 32),
            "cifar-10": (3, 32, 32),
            "cifar-100": (3, 32, 32),
            "svhn": (3, 32, 32),
        }[dataset_name]
    )


@torch.no_grad()
def generate_data(dataset_name: str, n_samples: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data for clustering.

    Args:
        dataset_name (str): Name of the dataset to generate. Possible values are:
            - "synth-2-clusters": Two clusters of points.
            - "synth-3-clusters": Three clusters of points.
            - "synth-9-clusters": Nine clusters of points.
            - "synth-2-moons": Two half-moon shapes.
            - "synth-circles": Two circles.
            - "synth-aniso": Anisotropicly distributed data.
            - "synth-checkerboard": Checkerboard pattern.
            - "synth-checkerboard-4": Four-cluster checkerboard pattern.
        n_samples (int): Number of samples to generate.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the generated data and labels.
    """
    tag = dataset_name.replace("synth-", "")
    if tag == "2-clusters":
        centers = [[0.0, 0.0], [0.5, 0.5]]
        cluster_stds = 0.1
        data, y = datasets.make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=centers,
            cluster_std=cluster_stds,
            random_state=0,
        )

    elif tag == "3-clusters":
        centers = [[0.0, 0.0], [0.5, 0.5], [0.5, 0.0]]
        cluster_stds = 0.05
        data, y = datasets.make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=centers,
            cluster_std=cluster_stds,
            random_state=0,
        )
    elif tag == "9-clusters":
        centers = [
            [0.0, 0.0],
            [0.5, 0.5],
            [0.5, 0.0],
            [0.0, 0.5],
            [0.5, 1.0],
            [1.0, 0.5],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
        cluster_stds = 0.1
        data, y = datasets.make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=centers,
            cluster_std=cluster_stds,
            random_state=0,
        )
    elif tag == "2-moons":
        data, y = datasets.make_moons(n_samples=n_samples, noise=0.3, random_state=0)

    elif tag == "circles":
        data, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)

    elif tag == "aniso":
        # Anisotropicly distributed data
        X, y = datasets.make_blobs(
            n_samples=n_samples,
            cluster_std=0.5,
            random_state=0,
            centers=[[-1, 0.5], [0.5, 0.5]],
        )
        transformation = [[0.5, -0.2], [-0.2, 0.4]]
        X_aniso = np.dot(X, transformation)
        data = X_aniso
    elif tag == "checkerboard":
        # X, Y = sklearn.datasets.make_checkerboard(shape=(2,4), n_clusters=2)
        centers = []
        for i in range(3):
            for j in range(3):
                centers.append([i, j])

        data, y = datasets.make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.2)

        y = y % 2

    elif tag == "checkerboard-4":
        # X, Y = sklearn.datasets.make_checkerboard(shape=(2,4), n_clusters=2)
        # centers = []
        # for i in range(2):
        #     for j in range(2):
        #         centers.append([i, j])

        centers = [[0, 0], [1, 1]]

        data_a, _ = datasets.make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.2)

        centers = [[0, 1], [1, 0]]

        data_b, _ = datasets.make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.2)
        data = np.concatenate([data_a, data_b], axis=0)
        y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)], axis=0)

    else:
        raise ValueError(f"Invalid synthetic dataset name: {tag}.")

    # Convert to tensors
    data = torch.from_numpy(data).float()
    labels = torch.from_numpy(y).long()

    # Squash data to N(0, 1)
    data = data - data.mean(dim=0)
    data = data / data.std(dim=0)

    return data, labels


def get_datasets(dataset_name: str, data_dir: str, full_train_set: bool) -> Tuple[Dataset, Optional[Dataset], Dataset]:
    """
    Get the specified dataset.


    Args:
        dataset_name (str): Dataset name.
        data_dir (str): Dataset root dir.
        full_train_set (bool): If True, the full training set is used. If False, a validation set is split off.

    Returns:
        Dataset: Dataset.
    """

    # Get the image size (assumes quadratic images)
    shape = get_data_shape(dataset_name)

    # Compose image transformations
    transform_train = v2.Compose(
        [
            v2.Resize(
                size=(shape.height, shape.width),
            ),
            v2.ToTensor(),
        ]
    )
    transform_test = v2.Compose(
        [
            v2.Resize(
                size=(shape.height, shape.width),
            ),
            v2.ToTensor(),
        ]
    )

    kwargs_train = dict(root=data_dir, download=True, transform=transform_train)
    kwargs_test = dict(root=data_dir, download=True, transform=transform_test)

    # Custom split generator with fixed seed
    split_generator = torch.Generator().manual_seed(0)

    # Set to none and check after if/else cases if its still none (unset, not defined by dataset)
    dataset_val = None

    # Select the datasets
    if "synth" in dataset_name:
        # Train
        data, labels = generate_data(dataset_name, n_samples=10000)
        dataset_train = torch.utils.data.TensorDataset(data, labels)

        # Val
        data, labels = generate_data(dataset_name, n_samples=5000)
        dataset_val = torch.utils.data.TensorDataset(data, labels)

        # Test
        data, labels = generate_data(dataset_name, n_samples=5000)
        dataset_test = torch.utils.data.TensorDataset(data, labels)

    elif dataset_name == "mnist" or dataset_name == "mnist-32":
        transform_train.transforms.append(v2.Normalize(mean=[0.1307], std=[0.3081]))
        # transform_train.transforms.append(v2.RandAugment())
        transform_test.transforms.append(v2.Normalize(mean=[0.1307], std=[0.3081]))

        dataset_train = MNIST(**kwargs_train, train=True)
        dataset_test = MNIST(**kwargs_test, train=False)

    elif dataset_name == "fmnist" or dataset_name == "fmnist-32":
        transform_train.transforms.append(v2.Normalize([0.5], [0.5]))
        transform_train.transforms.append(v2.RandAugment())
        # transform_train.transforms.append(v2.RandomHorizontalFlip())
        transform_test.transforms.append(v2.Normalize([0.5], [0.5]))

        dataset_train = FashionMNIST(**kwargs_train, train=True)
        dataset_test = FashionMNIST(**kwargs_test, train=False)

    elif dataset_name == "cifar-10":
        # transform_train.transforms.insert(1, v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10))
        transform_train.transforms.insert(1, v2.RandAugment())
        transform_train.transforms.append(
            v2.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            )
        )
        transform_test.transforms.append(
            v2.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            )
        )
        dataset_train = CIFAR10(**kwargs_train, train=True)
        dataset_test = CIFAR10(**kwargs_test, train=False)

    elif dataset_name == "cifar-2":
        # transform_train.transforms.insert(1, v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10))

        transform_train.transforms.append(
            v2.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            )
        )
        transform_test.transforms.append(
            v2.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            )
        )
        dataset_train = CIFAR10(**kwargs_train, train=True)
        dataset_test = CIFAR10(**kwargs_test, train=False)

        def make_binary(ds):
            mask = (np.array(ds.targets) == 0) + (np.array(ds.targets) == 1)
            ds.targets = np.array(ds.targets)[mask].tolist()
            ds.data = ds.data[mask]

        make_binary(dataset_train)
        make_binary(dataset_test)

    elif dataset_name == "cifar-100":
        transform_train.transforms.insert(1, v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10))

        transform_train.transforms.append(
            v2.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            )
        )
        transform_test.transforms.append(
            v2.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            )
        )
        dataset_train = CIFAR100(**kwargs_train, train=True)
        dataset_test = CIFAR100(**kwargs_test, train=False)

    elif dataset_name == "svhn":
        transform_train.transforms.insert(1, v2.AutoAugment(v2.AutoAugmentPolicy.SVHN))
        transform_train.transforms.append(v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        transform_test.transforms.append(v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

        # Load train and extra
        dataset_train = SVHN(**kwargs_train, split="train")

        dataset_test = SVHN(**kwargs_test, split="test")

        # Merge train and extra into train
        # dataset_extra = SVHN(**kwargs, split="extra")
        # dataset_train = ConcatDataset([dataset_train, dataset_extra])

    else:
        raise Exception(f"Unknown dataset: {dataset_name}")

    if dataset_val is None:
        N = len(dataset_train.data)

        if not full_train_set:
            N_train = round(N * 0.9)
            N_val = N - N_train
            lenghts = [N_train, N_val]
            dataset_train, dataset_val = random_split(dataset_train, lengths=lenghts, generator=split_generator)
        else:
            dataset_val = None

    return dataset_train, dataset_val, dataset_test


def build_dataloader(
    dataset_name: str,
    data_dir: str,
    num_workers: int,
    batch_size: int,
    full_train_set: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Builds and returns three data loaders for training, validation, and testing.

    Args:
        dataset_name (str): Name of the dataset to use.
        data_dir (str): Directory where the dataset is stored.
        num_workers (int): Number of worker threads to use for loading data.
        batch_size (int): Number of samples per batch.
        full_train_set (bool): If True, the full training set is used. If False, a validation set is split off.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple of three data loaders for training, validation, and testing.
    """
    # Get dataset objects
    dataset_train, dataset_val, dataset_test = get_datasets(
        dataset_name=dataset_name, data_dir=data_dir, full_train_set=full_train_set
    )

    # Build data loader
    loader_train = make_dataloader_from_dataset(
        dataset_train, num_workers=num_workers, shuffle=True, batch_size=batch_size
    )
    if not full_train_set:
        loader_val = make_dataloader_from_dataset(
            dataset_val, num_workers=num_workers, shuffle=False, batch_size=batch_size
        )
    else:
        loader_val = None
    loader_test = make_dataloader_from_dataset(
        dataset_test, num_workers=num_workers, shuffle=False, batch_size=batch_size
    )
    return loader_train, loader_val, loader_test


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = os.getpid() + int(datetime.now().strftime("%S%f")) + int.from_bytes(os.urandom(2), "big")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_reset_seed(worker_id: int):
    """Initialize the worker by settign a seed depending on the worker id.

    Args:
        worker_id (int): Unique worker id.
    """
    initial_seed = torch.initial_seed() % 2**31
    seed_all_rng(initial_seed + worker_id)


def make_dataloader_from_dataset(dataset: Dataset, num_workers: int, shuffle: bool, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_reset_seed,
        # multiprocessing_context="fork",  # Necessary for hydra multiprocessing runs
    )


def vis_student_synth_data(cfg, dataset_samples_val, logger_wandb, model_student):
    """
    Visualize synthetic data and decision boundaries.
    Args:
        cfg: Global configuration object.
        dataset_samples_val: Samples validation dataset.
        logger_wandb: WandbLogger object.
        model_student: Student model.
    """
    data, labels = generate_data(dataset_name=cfg.data.dataset, n_samples=1000)
    # Plot student boundary + data
    plot_data_and_decision_boundary(
        pred_func=model_student.predict,
        data=data,
        labels=labels,
        logger_wandb=logger_wandb,
        log_key="decision-boundary/student-data",
        device=model_student.device,
        x_min=-2.0,
        x_max=2.0,
        y_min=-2.0,
        y_max=2.0,
    )
    # Create temporary dataloader to sample 1K samples
    tmp_loader = make_dataloader_from_dataset(
        dataset=dataset_samples_val,
        num_workers=cfg.data.num_workers,
        batch_size=1000,
        shuffle=True,
    )
    samples_data, samples_labels = next(iter(tmp_loader))
    samples_data = samples_data.view(-1, 2)

    # Plot student boundary + samples
    plot_data_and_decision_boundary(
        pred_func=model_student.predict,
        data=samples_data,
        labels=samples_labels,
        logger_wandb=logger_wandb,
        log_key="decision-boundary/student-samples",
        x_min=-2.0,
        x_max=2.0,
        y_min=-2.0,
        y_max=2.0,
        device=model_student.device,
    )

import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms.v2 import functional as F


def plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

if __name__ == "__main__":

    train, val, test = get_datasets("svhn", data_dir="~/data", full_train_set=False)
    num_cols = 8
    num_rows = 8
    batch_size = num_cols
    dataloader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    imgs = []
    for i, (data, _) in enumerate(dataloader):
        imgs.append([img for img in data])
        if i == num_rows:
            break
    plot(imgs)
    plt.show()
    exit()


    data, label = generate_data(dataset_name="synth-checkerboard-4")

    from matplotlib import pyplot as plt

    # Plot data
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], c=label, cmap="bwr")
    plt.show()
