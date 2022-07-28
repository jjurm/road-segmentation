import os
import re

import albumentations as A
import cv2
import torch
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import functional as F

import utils as U
from configuration import CONSTANTS as C
from configuration import Configuration

stats = {
    'satimage'      :{'mean': [0.5161861897802257, 0.5216024102591976, 0.5315499204779472],
                        'std': [0.20524717537809348, 0.18630414534875536, 0.16632684150821872]},
    'boston'        :{'mean': [0.4564394591183637, 0.48059993284228936, 0.5224395830585711],
                        'std': [0.190038375170637,  0.17479450482743458, 0.16034113990076163]},
    'chicago'       :{'mean': [0.540109824511179, 0.5564791241094916, 0.5773300466334271],
                        'std': [0.2083955801686329, 0.1938503740257733, 0.189721605828102]},
    'houston':      {'mean': [0.4754334580352817, 0.4643549813812867, 0.4545077189830945],
                        'std': [0.1936710455402882, 0.1840672446845163, 0.18752556314166813]},
    'los_angeles':  {'mean': [0.5060523468395755, 0.5086963032043677, 0.5306354253020944],
                        'std': [0.16250931993546736, 0.15756912015440072, 0.14563739753468125]},
    'philadelphia': {'mean': [0.24759123901828473, 0.24728620370667695, 0.23709403991506356],
                        'std': [0.13198970810102156, 0.1293205432891552, 0.12462120945616202]},
    'phoenix':      {'mean': [0.41714666237510517, 0.404390573325625, 0.45269454896723194],
                        'std': [0.16299689577279053, 0.15103097161295354, 0.15956081921311085]},
    'san_francisco':{'mean': [0.43643431981511505, 0.43718305993771217, 0.4349842392333262],
                        'std': [0.16200691066633302, 0.15920377881429207, 0.14871560743571385]},
}


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

        it = self.transform(**it)

        if self.config.normalize:
            filepath = os.path.split(self.fnames_im[item])[0]
            basename = '_'.join(filepath.split('_')[:-1]) 
            it['image'] = F.normalize(stats[basename]['mean'], stats[basename]['std'])
        return it

    def __len__(self):
        return len(self.fnames_im)
