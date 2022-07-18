from typing import Iterable
from configuration import Configuration
from models.BaseModel import BaseModel
import torch
from torch import nn
import timm

class ResNetEncoder(BaseModel):
    def __init__(self, config: Configuration):
        super().__init__(config)

        if config.model_out == 'pixels':
            raise RuntimeError('ResNetEncoder only supports patchwise predictions.')

        name = config.model.lower()
        self.resnet = timm.create_model(name)

        # change stride of first convolution from 2 to 1 
        if isinstance(self.resnet.conv1, Iterable):
            self.resnet.conv1[0].stride = (1,1)
        else:
            self.resnet.conv1.stride = (1,1)

        self.ch = self.resnet.feature_info.channels()
        self.head = nn.Sequential(nn.Conv2d(self.ch[-1], 1, 1), nn.Sigmoid())


    def forward(self, batch: torch.Tensor):
        n_samples, n_channels, in_size, in_size = batch.shape

        batch = self.resnet(batch)
        batch = self.head()

        return batch.reshape(n_samples, self.out_size, self.out_size) # remove channel dim

    