import lightning as L
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader


class BrainDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, img_size=138):
        super().__init__()

        # Initialize parameters
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size


    def setup(self, stage=None):
        # Load your datasets here and split into train/val/test
        # Transformation for training data + data augmentation
        img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])

        # training dataset
        self.train_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, "train"),
            transform=img_transform
        )

        # test dataset
        self.val_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, "val"),
            transform=img_transform
        )

    def train_dataloader(self):
        # Return the training dataloader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        # Return the validation dataloader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

    def test_dataloader(self):
        # Return the test dataloader
        pass