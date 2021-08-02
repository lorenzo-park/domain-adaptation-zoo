import timm

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LitTimm(pl.LightningModule):
  def __init__(self, config_model):
    super().__init__()

    self.feature_extractor = timm.create_model(
        config_model.backbone,
        pretrained=False,
        num_classes=0
    )

    in_features = self.get_in_features(config_model.backbone)

    self.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, config_model.classes),
    )

    self.model = nn.Sequential(
      self.feature_extractor,
      self.classifier,
    )

    self.train_accuracy = pl.metrics.Accuracy()
    self.val_accuracy_src = pl.metrics.Accuracy()
    self.val_accuracy_tgt = pl.metrics.Accuracy()
    self.test_accuracy_src = pl.metrics.Accuracy()
    self.test_accuracy_tgt = pl.metrics.Accuracy()

  def training_step(self, batch, batch_idx):
    inputs, targets = batch["src"]

    outputs = self.model(inputs)
    loss = F.cross_entropy(outputs, targets)

    self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
    self.train_accuracy(outputs, targets)

    return loss

  def training_epoch_end(self, _):
    self.log("train_acc", self.train_accuracy.compute(),
             prog_bar=True, logger=True, sync_dist=True)

  def validation_step(self, batch, batch_idx):
    # Evaluate on source dataset
    inputs, targets = batch["src"]

    outputs = self.model(inputs)
    loss = F.cross_entropy(outputs, targets)

    self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
    self.val_accuracy_src(outputs, targets)

    # Evaluate on target dataset
    inputs, targets = batch["tgt"]

    outputs = self.model(inputs)
    loss = F.cross_entropy(outputs, targets)

    self.val_accuracy_tgt(outputs, targets)

    return loss

  def validation_epoch_end(self, _):
    self.log("val_acc_src", self.val_accuracy_src.compute(),
             prog_bar=True, logger=True, sync_dist=True)
    self.log("val_acc_tgt", self.val_accuracy_tgt.compute(),
             prog_bar=True, logger=True, sync_dist=True)

  def test_step(self, batch, batch_idx):
    # Evaluate on source dataset
    inputs, targets = batch["src"]

    outputs = self.model(inputs)
    loss = F.cross_entropy(outputs, targets)

    self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
    self.test_accuracy_src(outputs, targets)

    # Evaluate on target dataset
    inputs, targets = batch["tgt"]

    outputs = self.model(inputs)
    loss = F.cross_entropy(outputs, targets)

    self.test_accuracy_tgt(outputs, targets)

    return loss

  def test_epoch_end(self, _):
    self.log("test_acc_src", self.test_accuracy_src.compute(),
             prog_bar=True, logger=True, sync_dist=True)
    self.log("test_acc_tgt", self.test_accuracy_tgt.compute(),
             prog_bar=True, logger=True, sync_dist=True)

  def get_in_features(self, backbone):
    if "resnet50" == backbone:
      return 2048
    if "densenet121" == backbone:
      return 1024