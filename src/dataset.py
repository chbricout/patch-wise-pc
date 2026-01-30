import os

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class PTDataset(Dataset):
    def __init__(self, pt_path, colour_transform=None, resolution=64):
        data = torch.load(pt_path, mmap=True)
        if isinstance(data, dict):
            self.images = data["image"].view(
                -1, 3, resolution, resolution
            )  # (N,H,W,C) uint8 or (N,C,H,W)
        else:
            self.images = data.view(-1, 3, resolution, resolution)
        transforms_list = [
            transforms.Lambda(lambda x: x.long()),
        ]

        if colour_transform is not None:
            transforms_list.append(colour_transform)
        self.transform = transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)  # apply your pipeline

        return img


class MNIST(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, colour_transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (255 * x).int()),
            ]
        )

    def setup(self, stage: str):
        self.mnist_test = datasets.MNIST(
            self.data_dir, train=False, transform=self.transform, download=True
        )
        mnist_full = datasets.MNIST(
            self.data_dir, train=True, transform=self.transform, download=True
        )
        train_num = int(len(mnist_full) * 0.75)
        val_num = int(len(mnist_full) - train_num)

        self.mnist_train, self.mnist_val = random_split(
            mnist_full,
            [train_num, val_num],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class CIFAR(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, colour_transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (255 * x).int()),
        ]
        if colour_transform is not None:
            self.transform.append(colour_transform)
        self.transform = transforms.Compose(self.transform)

    def setup(self, stage: str):
        self.cifar_test = datasets.CIFAR10(
            self.data_dir, train=False, transform=self.transform, download=True
        )
        cifar_full = datasets.CIFAR10(
            self.data_dir, train=True, transform=self.transform, download=True
        )
        train_num = int(len(cifar_full) * 0.75)
        val_num = int(len(cifar_full) - train_num)
        self.cifar_train, self.cifar_val = random_split(
            cifar_full,
            [train_num, val_num],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)


class CelebA(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, colour_transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset_dir = os.path.join(data_dir, "celeba")

        self.colour_transform = colour_transform

    def setup(self, stage: str):
        self.celeba_test = PTDataset(
            os.path.join(self.dataset_dir, "celeba_test.pt"), self.colour_transform
        )
        self.celeba_train = PTDataset(
            os.path.join(self.dataset_dir, "celeba_train.pt"), self.colour_transform
        )
        self.celeba_val = PTDataset(
            os.path.join(self.dataset_dir, "celeba_valid.pt"), self.colour_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.celeba_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=3,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.celeba_val, batch_size=self.batch_size, pin_memory=True, num_workers=3
        )

    def test_dataloader(self):
        return DataLoader(
            self.celeba_test, batch_size=self.batch_size, pin_memory=True, num_workers=3
        )


class ImageNet(LightningDataModule):
    def __init__(
        self, data_dir: str, batch_size: int = 32, colour_transform=None, resolution=64
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset_dir = os.path.join(data_dir, "imagenet")
        self.colour_transform = colour_transform
        self.resolution = resolution

    def setup(self, stage: str):
        imnet_full = PTDataset(
            os.path.join(self.dataset_dir, f"imagenet{self.resolution}_train.pt"),
            self.colour_transform,
            resolution=self.resolution,
        )
        train_num = int(len(imnet_full) * 0.75)
        val_num = int(len(imnet_full) - train_num)

        self.imnet_train, self.imnet_val = torch.utils.data.random_split(
            imnet_full, [train_num, val_num], torch.Generator().manual_seed(42)
        )
        self.imnet_test = PTDataset(
            os.path.join(self.dataset_dir, f"imagenet{self.resolution}_val.pt"),
            self.colour_transform,
            resolution=self.resolution,
        )

        first_sample = self.imnet_train[2]
        print(first_sample.shape)
        print(first_sample.dtype)

    def train_dataloader(self):
        return DataLoader(
            self.imnet_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=3,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.imnet_val,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=3,
        )

    def test_dataloader(self):
        return DataLoader(
            self.imnet_test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=3,
        )
