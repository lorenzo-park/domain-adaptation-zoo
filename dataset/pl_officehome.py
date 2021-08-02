"""
CREDIT: https://github.com/thuml/Transfer-Learning-Library/blob/dev/common/vision/datasets/officehome.py
"""
from typing import Optional
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import ConcatDataset

import os

import pytorch_lightning as pl

from .imagelist import ImageList
from dataset.util import download as download_data, check_exits
from utils.aug import get_transforms_officehome


class OfficeHomeDataModule(pl.LightningDataModule):
  def __init__(self, root: str, batch_size: int, num_workers: int,
               src_task: str,
               tgt_task: Optional[str] = None,
               download: Optional[bool] = False):
    super().__init__()

    self.root = os.path.join(root, "data", "officehome")
    self.src_task = src_task
    self.tgt_task = tgt_task
    self.download = download
    self.transform = get_transforms_officehome()

    self.batch_size = batch_size
    self.num_workers = num_workers

    self.download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/ca3a3b6a8d554905b4cd/?dl=1"),
        ("Art", "Art.tgz", "https://cloud.tsinghua.edu.cn/f/4691878067d04755beab/?dl=1"),
        ("Clipart", "Clipart.tgz", "https://cloud.tsinghua.edu.cn/f/0d41e7da4558408ea5aa/?dl=1"),
        ("Product", "Product.tgz", "https://cloud.tsinghua.edu.cn/f/76186deacd7c4fa0a679/?dl=1"),
        ("Real_World", "Real_World.tgz", "https://cloud.tsinghua.edu.cn/f/dee961894cc64b1da1d7/?dl=1")
    ]
    self.image_list = {
        "Ar": "image_list/Art.txt",
        "Cl": "image_list/Clipart.txt",
        "Pr": "image_list/Product.txt",
        "Rw": "image_list/Real_World.txt",
    }
    self.CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
                    'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
                    'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
                    'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
                    'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
                    'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
                    'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']

  def prepare_data(self):
    assert self.src_task in self.image_list

    self.data_list_file_src = os.path.join(
        self.root, self.image_list[self.src_task])

    if self.tgt_task:
      assert self.tgt_task in self.image_list
      self.data_list_file_tgt = os.path.join(
          self.root, self.image_list[self.tgt_task])

    if self.download:
      list(map(lambda args: download_data(self.root, *args), self.download_list))
    else:
      list(map(lambda file_name, _: check_exits(
          self.root, file_name), self.download_list))

  def setup(self, stage: Optional[str] = None):
    if stage in (None, 'fit'):

      self.trainset_src = ImageList(self.root, self.CLASSES, "train",
                                 data_list_file=self.data_list_file_src,
                                 transform=self.transform["src"])

      self.valset_src = ImageList(self.root, self.CLASSES, "val",
                               data_list_file=self.data_list_file_src,
                               transform=self.transform["test"])

      if self.tgt_task:
        self.trainset_tgt = ImageList(self.root, self.CLASSES, "all",
                                  data_list_file=self.data_list_file_tgt,
                                  transform=self.transform["tgt"])

    if stage in (None, 'test'):
      if self.tgt_task:
        self.testset = ImageList(self.root, self.CLASSES, "all",
                                  data_list_file=self.data_list_file_tgt,
                                  transform=self.transform["test"])
      else:
        self.testset = ImageList(self.root, self.CLASSES, "val",
                                  data_list_file=self.data_list_file_src,
                                  transform=self.transform["test"])

  def train_dataloader(self):
    return CombinedLoader({
        "src": DataLoader(self.trainset_src, batch_size=self.batch_size,
                          shuffle=True, pin_memory=True, num_workers=self.num_workers, drop_last=True),
        "tgt": DataLoader(self.trainset_tgt, batch_size=self.batch_size,
                          shuffle=True, pin_memory=True, num_workers=self.num_workers, drop_last=True),
    }, "max_size_cycle")

  def val_dataloader(self):
    if self.tgt_task:
      return DataLoader(self.trainset_tgt, batch_size=self.batch_size,
                        num_workers=self.num_workers, drop_last=True)
    else:
      return DataLoader(self.valset_src, batch_size=self.batch_size,
                        num_workers=self.num_workers, drop_last=True)

  def test_dataloader(self):
    if self.tgt_task:
      return DataLoader(self.trainset_tgt, batch_size=self.batch_size,
                        num_workers=self.num_workers, drop_last=True)
    else:
      return DataLoader(self.valset_src, batch_size=self.batch_size,
                        num_workers=self.num_workers, drop_last=True)

