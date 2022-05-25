import cv2
from matplotlib.transforms import Transform
import torch
from albumentations import Compose as ComposeTransforms
from albumentations.pytorch.transforms import ToTensorV2 
from albumentations.core.transforms_interface import DualTransform, NoOp

import utils as U
from configuration import CONSTANTS as C
from configuration import Configuration


class SatelliteData(torch.utils.data.Dataset):

    def __init__(self, dataset:str, config:Configuration, train=False, transform:DualTransform=NoOp) -> None:
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.train = train

        # Prepare albumentation transform
        transform = ComposeTransforms([transform, ToTensorV2], p=1)
        self.transform_train = transform()
        self.transform_test = ToTensorV2()

        self.fnames_im = U.get_filenames(dataset, 'images')
        if train:
            self.fnames_gt = U.get_filenames(dataset, 'groundtruth')


    def __getitem__(self, item):
        
        if self.train:
            im = cv2.imread(self.fnames_im[item])
            gt = cv2.imread(self.fnames_gt[item], cv2.IMREAD_GRAYSCALE)
            t = self.transform_train(image=im, mask=gt)
            return t['image'], t['mask']
        
        else:
            im = cv2.imread(self.fnames_im[item])
            t = self.transform_test(image=im)
            return t['image']


    def __len__(self):
        return len(self.fnames_im)
