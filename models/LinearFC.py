import torch
from configuration import CONSTANTS as C
from configuration import Configuration
from torch import nn
from torch.nn import functional as F

from models.BaseModel import BaseModel


class LinearFC(BaseModel):
    '''
    This is a simple logistic regression model. It provides basic 
    implementations to demonstrate how more advanced models can be built.
    '''

    def __init__(self, config:Configuration):
        super().__init__(config)

        # prepare model
        self.in_dim = 3 * C.IMG_SIZE * C.IMG_SIZE
        self.out_dim = self.out_size * self.out_size

        if self.config.model_out == 'pixel':
            raise RuntimeError('FC would have 3*400^4 parameters, '
            + 'i.e. ~307GB.. please use patchwise predictions instead.')

        self.linear = nn.Linear(in_features=self.in_dim, out_features=self.out_dim)


    def forward(self, batch:torch.Tensor):
        n_samples, n_channels, in_width, in_height = batch.shape

        # forward through linear layer
        batch = batch.reshape(n_samples, -1)
        batch = self.linear(batch)
        batch = torch.sigmoid(batch)

        return batch.reshape(n_samples, 1, self.out_size, self.out_size)
    
