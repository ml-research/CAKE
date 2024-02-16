from enum import Enum
from typing import Tuple

import numpy as np

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR
from rtpt import RTPT
from torch import nn
from torch.nn import functional as F

from src.models.base import make_model

# Translate the dataloader index to the dataset name
DATALOADER_ID_TO_SET_NAME = {0: "train_real", 1: "val_real", 2: "test_real"}
DATALOADER_ID_TO_SET_NAME_WITHOUT_VAL = {0: "train_real", 1: "test_real"}


class ModelType(str, Enum):
    """Simple Enum to differentiate between the teacher and student model."""

    TEACHER = "teacher"
    STUDENT = "student"


def nll_loss(log_posterior, targets):
    """
    NLL loss with smooth targets.

    Args:
        log_posterior: Log-Posterior of shape [N, C].
        targets: Targets of shape [N, C] or [N,].

    Returns:
        Loss.

    """
    if targets.dim() == 1:
        loss = F.nll_loss(log_posterior, targets)
    elif targets.dim() == 2:
        loss = -(targets * log_posterior).sum(dim=1).mean()
    else:
        raise ValueError("Invalid smooth_targets dimension, expected 1 or 2 but got " + str(targets.dim()))
    return loss


class DiscriminativeModel(LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        learning_rate: float,
        weight_decay: float,
        epochs: int,
        model_cfg: DictConfig,
        steps_per_epoch: int,
        model_type: ModelType,
    ):
        """
        Construct a discriminative model.

        Args:
            cfg: Config.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            epochs: Number of epochs.
            model_cfg: Model config.
            steps_per_epoch: Number of steps per epoch.
            model_type: Type of model (teacher or student).

        """
        super(DiscriminativeModel, self).__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.model = make_model(cfg, model_cfg=model_cfg.model)
        self.steps_per_epoch = steps_per_epoch
        self.model_type = model_type
        self.cfg = cfg
        self.model_cfg = model_cfg
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()

        # Setup RTPT
        self.rtpt = RTPT(
            name_initials="SB",
            experiment_name=f"CAKE_{self.model_type}" + ("_" + str(cfg.env.tag) if cfg.env.tag else ""),
            max_iterations=epochs + 1,
        )

    def configure_optimizers(self):
        optim_type = self.model_cfg.optim.type
        scheduler_type = self.model_cfg.optim.scheduler.type

        if optim_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        elif optim_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif optim_type == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(
                f"Value of model_cfg.optim.type must be one of [sgd, adam, adagrad, adamw] but was" f" {optim_type}."
            )

        if scheduler_type == "1cycle":
            scheduler = {
                "scheduler": OneCycleLR(
                    optimizer,
                    max_lr=self.learning_rate,
                    total_steps=self.epochs * self.steps_per_epoch
                    + 1,  # +1 b/c 1cycle has a bug in its last step where it upticks the lr again
                    div_factor=self.model_cfg.optim.scheduler.onecycle.div_factor,
                    final_div_factor=self.model_cfg.optim.scheduler.onecycle.final_div_factor,
                ),
                "interval": "step",
            }
        elif scheduler_type == "multistep":
            ms_p = self.model_cfg.optim.scheduler.multistep.milestones
            ms = [round(float(p) * self.epochs) for p in ms_p]
            scheduler = MultiStepLR(optimizer, milestones=ms)
        else:
            raise ValueError(f"Invalid scheduler_type: {scheduler_type}")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_start(self) -> None:
        self.rtpt.start()

    def on_train_epoch_end(self) -> None:
        self.rtpt.step()

    def training_step(self, train_batch, batch_idx):
        loss, accuracy = self._get_loss_and_accuracy(train_batch)
        self.log(f"train/{self.model_type}/accuracy", accuracy, prog_bar=True)
        self.log(f"train/{self.model_type}/loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        # Add data/samples tag, depending on the dataloader idx for the student
        if self.model_type == ModelType.STUDENT:
            dataloader_tag = ["fake", "real"][dataloader_idx]
        else:
            dataloader_tag = "real"

        loss, accuracy = self._get_loss_and_accuracy(val_batch)

        # Log accuracy and loss
        self.log(
            f"val/{self.model_type}/accuracy_{dataloader_tag}",
            accuracy,
            add_dataloader_idx=False,
            prog_bar=True,
        )
        self.log(f"val/{self.model_type}/loss_{dataloader_tag}", loss, add_dataloader_idx=False)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, accuracy = self._get_loss_and_accuracy(batch)
        if self.cfg.data.full_train_set:
            set_name = DATALOADER_ID_TO_SET_NAME_WITHOUT_VAL[dataloader_idx]
        else:
            set_name = DATALOADER_ID_TO_SET_NAME[dataloader_idx]
        self.log(f"test/{self.model_type}/accuracy_{set_name}", accuracy, add_dataloader_idx=False)

    def _get_loss_and_accuracy(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss and accuracy of batch.
        Args:
            batch: Batch of data.

        Returns:
            Tuple of (loss, accuracy).
        """
        data, labels = batch
        outputs = self.model(data)

        # Compute loss
        if "svm" in self.model_cfg.model.arch:
            loss = self.model.loss(outputs, labels)
        else:
            loss = self.criterion(outputs, labels)

        if labels.dim() == 1:
            y_true = labels
        elif labels.dim() == 2:
            y_true = labels.argmax(-1)
        else:
            raise ValueError(f"Invalid label dimension: {labels.dim()}")

        preds = outputs.argmax(-1)
        num_correct = (y_true == preds).sum()
        accuracy = num_correct / outputs.shape[0] * 100

        return loss, accuracy

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        return self.forward(data)
