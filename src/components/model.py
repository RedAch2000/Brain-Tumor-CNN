import lightning as L
import torch
import torch.nn as nn
from torchvision import models
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score


class BrainTumorModel(L.LightningModule):

    def __init__(self, lr=1e-4, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        # Model
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

        # Loss
        self.class_weights = class_weights
        self.criterion = None

        # Metrics (binary classification)
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")

        self.train_precision = Precision(task="binary")
        self.val_precision = Precision(task="binary")

        self.train_recall = Recall(task="binary")
        self.val_recall = Recall(task="binary")

        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")

    def setup(self, stage=None):
        if self.class_weights is not None:
            weights = self.class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.train_acc(preds, y)
        self.train_precision(preds, y)
        self.train_recall(preds, y)
        self.train_f1(preds, y)

        # Log
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        self.log("train_precision", self.train_precision)
        self.log("train_recall", self.train_recall)
        self.log("train_f1", self.train_f1)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.val_acc(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)

        # Log
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_precision", self.val_precision)
        self.log("val_recall", self.val_recall, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)