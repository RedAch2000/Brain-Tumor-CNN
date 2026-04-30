import lightning as L
from dataclasses import dataclass
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np


@dataclass
class DataTransformationConfig:
    data_dir: Path = Path("dataset/raw")
    batch_size: int = 16
    img_size: tuple = (128, 128)


class DataTransformationModule(L.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.config = DataTransformationConfig()

    def setup(self, stage=None):

        # training transforms
        train_transform = transforms.Compose([
            transforms.Resize(self.config.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])

        # validation transforms (only resizing and tensor conversion)
        val_transform = transforms.Compose([
            transforms.Resize(self.config.img_size),
            transforms.ToTensor(),
        ])

        # load WITHOUT transform first for splitting
        base_dataset = ImageFolder(root=self.config.data_dir)

        indices = np.arange(len(base_dataset))
        labels = np.array(base_dataset.targets)

        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.2,
            stratify=labels,
            random_state=42
        )

        # rebuild datasets with correct transforms
        train_dataset = ImageFolder(
            self.config.data_dir,
            transform=train_transform
        )

        val_dataset = ImageFolder(
            self.config.data_dir,
            transform=val_transform
        )

        self.train_dataset = Subset(train_dataset, train_idx)
        self.val_dataset = Subset(val_dataset, val_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )