import re

import augmentations as A
import cv2
import torch
from albumentations.pytorch.transforms import ToTensorV2

import utils as U
from configuration import CONSTANTS as C
from configuration import Configuration


class SatelliteData(torch.utils.data.Dataset):

    def __init__(self, dataset:str, config:Configuration, train=True, transform:A.DualTransform=A.NoOp()) -> None:
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.train = train
        self.transform = A.Compose([transform, U.ToFloatDual(), ToTensorV2()])

        # get dataset filnames
        self.fnames_im = U.get_filenames(dataset, 'images')
        if train:
            self.fnames_gt = U.get_filenames(dataset, 'groundtruth')

        # get dataset indices
        self.indices = [int(re.findall(r'\d+', f)[-1]) for f in self.fnames_im]


    def __getitem__(self, item):

        it = {}
        it['image'] = cv2.imread(self.fnames_im[item])
        it['image'] = cv2.cvtColor(it['image'], cv2.COLOR_BGR2RGB)

        if self.train:
            it['mask'] = cv2.imread(self.fnames_gt[item])
            it['mask'] = cv2.cvtColor(it['mask'], cv2.COLOR_BGR2GRAY)
            
            
        if not self.train: # for testset predictions
            it['idx'] = self.indices[item]

        return self.transform(**it)

    def __len__(self):
        return len(self.fnames_im)
