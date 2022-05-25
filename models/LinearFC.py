import torch
from configuration import CONSTANTS as C
from configuration import Configuration
from torch import nn
from torch.nn import functional as F

from BaseModel import BaseModel


class LinearFC(BaseModel):
    '''
    This is a simple logistic regression model. It provides basic 
    implementations to demonstrate how more advanced models can be built.
    '''

    def __init__(self, config:Configuration):
        super().__init__(config)

        # prepare model
        in_dim = 3 * C.IMG_SIZE * C.IMG_SIZE
        if self.config.model_out == 'pixels':
            out_dim = C.IMG_SIZE * C.IMG_SIZE
        else:
            out_dim = C.PATCH_SIZE * C.PATCH_SIZE

        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)


    def forward(self, batch:torch.Tensor):
        n_samples, n_channels, width, height = batch.shape

        # forward through linear layer
        batch = batch.reshape(n_samples, -1)
        batch = self.linear(batch)
        batch = F.sigmoid(batch)

        return batch.reshape(n_samples, width, height)
    
