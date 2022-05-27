import torch
from configuration import CONSTANTS as C
from configuration import Configuration
from torch import nn
from torch.nn import functional as F

from models.BaseModel import BaseModel


class LinearConv(BaseModel):
    '''
    This is a simple logistic regression model. It provides basic 
    implementations to demonstrate how more advanced models can be built.
    '''

    def __init__(self, config:Configuration):
        super().__init__(config)

        # prepare model
        if config.model_out == 'patches':
            self.kernel_size = C.PATCH_SIZE
        else:
            self.kernel_size = 1 

        # out is linear combination of one pixel or patch
        self.conv = nn.Conv2d(in_channels=3, 
                              out_channels=1, 
                              kernel_size=self.kernel_size,
                              stride=self.kernel_size)

    def forward(self, batch:torch.Tensor):
        n_samples, n_channels, in_size, in_size = batch.shape
        # forward through linear conv layer
        batch = self.conv(batch)
        batch = torch.sigmoid(batch)

        return batch.reshape(n_samples, self.out_size, self.out_size) # remove channel dim
    
