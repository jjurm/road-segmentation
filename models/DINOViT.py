from copy import deepcopy
from typing import Any, Dict, Iterable, List, OrderedDict

import timm
import torch
from configuration import Configuration, create_optimizer
from configuration import CONSTANTS as C
from timm.models.features import FeatureListNet
from torch import nn
from math import sqrt
from torch.nn import functional as F

from models.BaseModel import BaseModel


__supported_models__ = ['DINOViTs', 'DINOViTb']
class DINOViT(BaseModel):
    def __init__(self, config: Configuration):
        super().__init__(config)

        if config.model == 'DINOViTs':
            model_str = 'dino_vits16'
        elif config.model == 'DINOViTb':
            model_str = 'dino_vitb16'
        else:
            raise RuntimeError(f'{config.model} is not supported, please try one of {__supported_models__}')
        
        if config.model_out == 'pixel':
            raise RuntimeError('DINOViT only supports patchwise outputs')
            
        self.dino = torch.hub.load('facebookresearch/dino:main', model_str)
        self.head = nn.Conv2d(self.dino.num_features, 1, 1)
        self.out_channels = self.dino.num_features

    def configure_optimizers(self):
        params = list(self.head.parameters())
        t_opt, kwargs = create_optimizer(self.config)
        return t_opt(params, **kwargs)

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.config.freeze_epochs:
            t_opt, kwargs = create_optimizer(self.config)
            self.trainer.optimizers = [t_opt(self.parameters(), **kwargs)]
            #self.optimizers().param_groups.clear() # remove all param_groups
            #param_group = dict(params=self.parameters()) # add all parameters
            #self.optimizers().add_param_group(param_group)
        return 

    def forward(self, batch: torch.Tensor):
        n_samples, n_channels, in_size, in_size = batch.shape
        out_size = in_size // C.PATCH_SIZE 

        # adjust from here:
        # https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/vision_transformer.py#L209
        x = self.dino.prepare_tokens(batch)
        for blk in self.dino.blocks:
            x = blk(x)
        x = self.dino.norm(x)
        x = x[:, 1:, :]                 # n_samples, n_blocks, n_channels => remove [CLS] token
        x = x.transpose(1, 2)           # n_samples, n_channels, n_patches**2
        x = F.normalize(x, p=2, dim=1)  # n_samples, n_channels, n_patches**2

        # reshape batch sequence back to image embedding
        x = x.reshape([n_samples, -1, out_size, out_size])

        batch = self.head(x)
        return batch #batch.reshape(n_samples, 1, self.out_size, self.out_size)

    
