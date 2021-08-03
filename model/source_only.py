import timm
import torch
import torchmetrics

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LitTimm(pl.LightningModule):
  def __init__(self, config_backbone, config_training, num_classes, learning_rate=None):
    super().__init__()

    self.config_training = config_training

    if learning_rate:
      self.learning_rate = learning_rate
    else:
      self.learning_rate = self.config_training.lr

    in_features = self.get_in_features(config_backbone.name)

    self.feature_extractor = timm.create_model(
        config_backbone.name,
        pretrained=True,
        num_classes=0
    )
    self.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )

    self.train_accuracy = torchmetrics.Accuracy()
    self.val_accuracy = torchmetrics.Accuracy()
    self.test_accuracy = torchmetrics.Accuracy()

  def training_step(self, batch, batch_idx):
    inputs, targets = batch["src"]

    outputs = self.forward_cls(inputs)
    loss = F.cross_entropy(outputs, targets)

    self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
    self.train_accuracy(outputs, targets)

    return loss

  def training_epoch_end(self, _):
    self.log("train_acc", self.train_accuracy.compute(),
             prog_bar=True, logger=True, sync_dist=True)

  def validation_step(self, batch, batch_idx):
    # Evaluate on source dataset
    inputs, targets = batch

    outputs = self.forward_cls(inputs)
    loss = F.cross_entropy(outputs, targets)

    self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
    self.val_accuracy(outputs, targets)

    return loss

  def validation_epoch_end(self, _):
    self.log("val_acc", self.val_accuracy.compute(),
             prog_bar=True, logger=True, sync_dist=True)

  def test_step(self, batch, batch_idx):
    # Evaluate on source dataset
    inputs, targets = batch
    outputs = self.forward_cls(inputs)
    loss = F.cross_entropy(outputs, targets)

    self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
    self.test_accuracy(outputs, targets)

    return loss

  def test_epoch_end(self, _):
    self.log("test_acc", self.test_accuracy.compute(),
             prog_bar=True, logger=True, sync_dist=True)

  def configure_optimizers(self):
    model_params = [
        {
            "params": self.feature_extractor.parameters(),
            "lr": self.learning_rate * self.config_training.w_enc
        },
        {
            "params": self.classifier.parameters(),
            "lr": self.learning_rate * self.config_training.w_cls
        },
    ]
    if self.config_training.optimizer == "sgd":
      optimizer = torch.optim.SGD(
          model_params,
          lr=self.learning_rate,
          momentum=0.9,
          weight_decay=1e-5,
          nesterov=True,
      )
    elif self.config_training.optimizer == "adam":
      optimizer = torch.optim.Adam(
          model_params,
          lr=self.learning_rate,
      )
    elif self.config_training.optimizer == "adamw":
      optimizer = torch.optim.AdamW(
          model_params,
          lr=self.learning_rate,
      )

    return optimizer

  def forward_cls(self, x):
    x = self.feature_extractor(x)
    x = self.classifier(x)

    return x

  def get_in_features(self, backbone):
    if "resnet50" == backbone:
      return 2048
    if "densenet121" == backbone:
      return 1024
