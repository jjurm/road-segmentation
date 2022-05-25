import os

import torch
from albumentations.core.transforms_interface import DualTransform, NoOp
from PIL import Image

import utils as U
from configuration import CONSTANTS as C
from configuration import Configuration


class SatelliteData(torch.utils.data.Dataset):

    def __init__(self, dataset:str, config:Configuration, transform:DualTransform=NoOp) -> None:
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.transform = transform
        
        indices_im = U.get_dataindices(self.__resolve_path__('images', '*'))
        indices_gt = U.get_dataindices(self.__resolve_path__('groundtruth', '*'))

        if indices_im != indices_gt:
            raise RuntimeError('List of groundtruth and images are not coherent.')
        self.indices = indices_im


    def __resolve_path__(self, dir, item):
        idx = self.indices[item]
        return os.path.join(C.DATA, self.dataset, dir, f'satimage_{idx}.png')


    def __getitem__(self, item):
        im = Image.open(self.__resolve_path__('images', item))
        gt = Image.open(self.__resolve_path__('groundtruth', item))

        # apply albumentations dual transform
        t = self.transform(im=im, gt=gt)
        return t['im'], t['gt']


    def __len__(self):
        return len(self.indices)
