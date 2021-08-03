from torch.optim.lr_scheduler import _LRScheduler

import timm
import torch
import torchmetrics

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class GRL(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, alpha):
    ctx.alpha = alpha

    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
    output = grad_output.neg() * ctx.alpha

    return output, None


class DANNLRScheduler(_LRScheduler):
  def __init__(self, optimizer, multiplier, alpha, beta, total_steps):
    self.multiplier = multiplier
    self.alpha = alpha
    self.beta = beta
    self.total_steps = total_steps

    super(DANNLRScheduler, self).__init__(optimizer)

  def get_lr(self):
    current_iterations = self.optimizer._step_count
    p = float(current_iterations / self.total_steps)

    return [
        param_group["lr"] * self.multiplier / (1 + self.alpha * p) ** self.beta
        for param_group in self.optimizer.param_groups
    ]

    # for param_group in self.optimizers().param_groups:
    #   param_group["lr"] = \
    #       param_group["lr_init"] * self.config_training.lr / \
    #       (1 + self.config_training.alpha * p) ** self.config_training.beta


class LitDANN(pl.LightningModule):
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

    self.discriminator = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=1024),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(in_features=1024, out_features=1024),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(in_features=1024, out_features=2)
    )

    self.register_buffer("targets_d_src", torch.ones(
        config_training.batch_size).long())
    self.register_buffer("targets_d_tgt", torch.zeros(
        config_training.batch_size).long())

    self.train_accuracy = torchmetrics.Accuracy()
    self.val_accuracy = torchmetrics.Accuracy()
    self.test_accuracy = torchmetrics.Accuracy()

  def training_step(self, batch, batch_idx):
    self.on_train_epoch_start()
    inputs_src, targets_src = batch["src"]

    p = self.get_p()
    lambda_p = self.get_lambda_p(p)

    outputs_src, features_src = self.forward_cls(inputs_src)
    loss = F.cross_entropy(outputs_src, targets_src)

    self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
    self.train_accuracy(outputs_src, targets_src)

    inputs_tgt, _ = batch["tgt"]
    outputs_d_src, outputs_d_tgt = self.forward_dsc(
        features_src, inputs_tgt, lambda_p=lambda_p)
    loss += F.cross_entropy(outputs_d_src, self.targets_d_src)
    loss += F.cross_entropy(outputs_d_tgt, self.targets_d_tgt)

    return loss

  def training_epoch_end(self, _):
    self.log("train_acc", self.train_accuracy.compute(),
             prog_bar=True, logger=True, sync_dist=True)

  def validation_step(self, batch, batch_idx):
    # Evaluate on source dataset
    inputs, targets = batch

    outputs, _ = self.forward_cls(inputs)
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

    outputs, _ = self.forward_cls(inputs)
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
            "lr": self.learning_rate * self.config_training.w_enc,
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

    if self.config_training.lr_schedule:
      lr_scheduler = DANNLRScheduler(
        optimizer=optimizer, multiplier=0.1,
        alpha=self.config_training.alpha,
        beta=self.config_training.beta,
        total_steps=len(self.train_dataloader()) * self.config_training.epoch
      )
      return [optimizer], [lr_scheduler]
    else:
      return optimizer

  def forward_cls(self, x):
    features = self.feature_extractor(x)
    x = self.classifier(features)

    return x, features

  def forward_dsc(self, features_src, x_tgt, lambda_p):
    x_src = GRL.apply(features_src, lambda_p)
    x_src = self.discriminator(x_src)

    x_tgt = self.feature_extractor(x_tgt)
    x_tgt = GRL.apply(x_tgt, lambda_p)
    x_tgt = self.discriminator(x_tgt)

    return x_src, x_tgt

  def get_in_features(self, backbone):
    if "resnet50" == backbone:
      return 2048
    if "densenet121" == backbone:
      return 1024

  def get_p(self):
    current_iterations = self.global_step
    len_dataloader = len(self.train_dataloader())
    p = float(current_iterations /
              (self.config_training.epoch * len_dataloader))

    return p

  def get_lambda_p(self, p):
    lambda_p = 2. / (1. + np.exp(-self.config_training.gamma * p)) - 1

    return lambda_p
