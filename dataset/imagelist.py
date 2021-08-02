"""
CREDIT: https://github.com/thuml/Transfer-Learning-Library/tree/dev/dalib/vision/datasets
"""

from torch.utils.data import random_split
from torchvision.datasets.folder import default_loader
from typing import Optional, Callable, Tuple, Any, List

import os
import torchvision.datasets as datasets


class ImageList(datasets.VisionDataset):
  """A generic Dataset class for domain adaptation in image classification
  Parameters:
      - **root** (str): Root directory of dataset
      - **classes** (List[str]): The names of all the classes
      - **data_list_file** (str): File to read the image list from.
      - **transform** (callable, optional): A function/transform that  takes in an PIL image \
          and returns a transformed version. E.g, ``transforms.RandomCrop``.
      - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
  .. note:: In `data_list_file`, each line 2 values in the following format.
      ::
          source_dir/dog_xxx.png 0
          source_dir/cat_123.png 1
          target_dir/dog_xxy.png 0
          target_dir/cat_nsdf3.png 1
      The first value is the relative path of an image, and the second value is the label of the corresponding image.
      If your data_list_file has different formats, please over-ride `parse_data_file`.
  """

  def __init__(self, root: str, classes: List[str], split: str, data_list_file: str,
               transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
    super().__init__(root, transform=transform, target_transform=target_transform)
    self.split = split
    self.data = self.parse_data_file(data_list_file)
    self.classes = classes
    self.class_to_idx = {cls: idx
                         for idx, clss in enumerate(self.classes)
                         for cls in clss}
    self.loader = default_loader

  def __getitem__(self, index: int) -> Tuple[Any, int]:
    """
    Parameters:
        - **index** (int): Index
        - **return** (tuple): (image, target) where target is index of the target class.
    """
    path, target = self.data[index]
    img = self.loader(path)
    if self.transform is not None:
      img = self.transform(img)
    if self.target_transform is not None and target is not None:
      target = self.target_transform(target)
    return img, target

  def __len__(self) -> int:
    return len(self.data)

  def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
    """Parse file to data list
    Parameters:
        - **file_name** (str): The path of data file
        - **return** (list): List of (image path, class_index) tuples
    """
    with open(file_name, "r") as f:
      data_list = []
      for line in f.readlines():
        path, target = line.split()
        if not os.path.isabs(path):
          path = os.path.join(self.root, path)
        target = int(target)
        data_list.append((path, target))

    split_lengths = [
        len(data_list) - len(data_list) // 10,
        len(data_list) // 10
    ]

    data_list_file_src, data_list_file_src_val = random_split(
        data_list, split_lengths
    )

    if self.split == "train":
      return data_list_file_src
    elif self.split == "val":
      return data_list_file_src_val
    else:
      return data_list

  @property
  def num_classes(self) -> int:
    """Number of classes"""
    return len(self.classes)
