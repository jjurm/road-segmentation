from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from utils import Pix2Patch
from configuration import CONSTANTS as C
from configuration import Configuration, create_loss, create_optimizer


class BaseModel(pl.LightningModule):
    """A base class for neural networks that defines an interface and implements automatic
    handling of patch-wise and pixel-wise loss functions and metrics. By default, a model 
    is expected to ouput pixel-wise and patch-wise, depending on config.model_out. Please
    raise an error, if your model does not support one of the methods."""

    def __init__(self, config:Configuration):
        super().__init__()

        self.config = config
        self.loss = create_loss(config)

        if config.model_out == 'patch' and config.loss_in == 'pixel':
            raise RuntimeError(f'Invalid configuration: model_out=patch, loss_in=pixel.')

        # prepare dimensions:
        if self.config.model_out == 'patch':
            self.out_size = int(C.IMG_SIZE / C.PATCH_SIZE)

        elif self.config.model_out == 'pixel':
            self.out_size = C.IMG_SIZE

        # automatic pixel to patch transform by averaging 
        self.pix2patch = Pix2Patch(C.PATCH_SIZE)

        # output modes of model: for pixelwise model, also the patchwise outputs are tracked
        self.outmodes = ['patch', 'pixel'] if (config.model_out=='pixel') else ['patch']

        self.save_hyperparameters()


    def configure_optimizers(self):
        optimizer = create_optimizer(self, self.config)
        return optimizer

    def forward(self, batch:torch.Tensor) -> torch.Tensor:
        n_samples, n_channels, in_size, in_size = batch.shape
        raise NotImplementedError("Must be implemented by subclass.")
        return batch.reshape(n_samples, 1, self.out_size, self.out_size)


    def step(self, batch:Dict[str, torch.Tensor], batch_idx):
        images = batch['image']
        targets = batch['mask'].unsqueeze(1) # get channel dimension

        # forward through model to obtain probabilities
        probas = self(images)

        # nested dict for patchwise and pixelwise prediction as output
        out = dict([(mode, dict()) for mode in self.outmodes])      

        # model output might come pixel- or patchwise
        if self.config.model_out == 'pixel':
            out['pixel']['probas'] = probas # probas are pixelwise
            out['pixel']['targets'] = targets # targets are pixelwise
            out['patch']['probas'] = self.pix2patch(probas)
            out['patch']['targets'] = self.pix2patch(targets)
        else: 
            out['patch']['probas'] = probas # probas are patchwise
            out['patch']['targets'] = self.pix2patch(targets) # targets are patchwise
            

        # compute loss from pixel or patch with soft predictions/targets
        if self.config.loss_in == 'pixel':
            out['loss'] = self.loss(out['pixel']['probas'], out['pixel']['targets'])
        else:
            out['loss'] = self.loss(out['patch']['probas'], out['patch']['targets'])

        return out


    def training_step(self, batch:dict, batch_idx):
        out = self.step(batch, batch_idx)
        self.log('train/loss', out['loss'])
        return out
    

    def validation_step(self, batch:dict, batch_idx):
        out = self.step(batch, batch_idx)
        self.log('valid/loss', out['loss'])
        return out


    def apply_threshold(self, soft_labels, outmode:str) -> Tuple[torch.Tensor, torch.Tensor]:
        threshold = C.THRESHOLD if (outmode=='patch') else 0.5
        hard_labels = (soft_labels > threshold).float() # convert to hard labels
        return hard_labels


    def predict_step(self, batch:dict, batch_idx):
        images = batch['image']
        i_inds = batch['idx'] # image indices

        # get probabilities
        probas = self(images)
        if self.config.model_out == 'pixel': 
            probas = self.pix2patch(probas)

        # get predictions
        preds = self.apply_threshold(probas, 'patch').int().squeeze() # remove channels

        # get submission table
        rows = []
        for k in range(preds.shape[0]):
            for i in range(preds.shape[1]):
                for j in range(preds.shape[2]):
                    row = [i_inds[k], j*C.PATCH_SIZE, i*C.PATCH_SIZE, preds[k, i, j]]
                    rows.append(torch.tensor(row).unsqueeze(0))

        return torch.cat(rows, dim=0)
