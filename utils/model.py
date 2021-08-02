from pytorch_lightning.metrics.utils import get_num_classes
import torch.nn as nn

from source_only.model import LitTimm


def get_model(config, num_classes):
  if config.model_name == "source_only":
    return LitTimm(config.model, config.training, num_classes)


def init_weights(m):
  if type(m) == nn.Linear:
    nn.init.xavier_normal_(m.weight)
    nn.init.constant_(m.bias, 0)

  if type(m) == nn.BatchNorm1d:
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)
