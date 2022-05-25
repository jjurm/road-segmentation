
from numpy import indices
import pytorch_lightning as pl
import torch
import utils as U
from configuration import CONSTANTS as C
from configuration import Configuration, create_loss, create_optimizer
from torchmetrics import F1Score


class BaseModel(pl.LightningModule):
    """A base class for neural networks that defines an interface and implements a few common functions."""

    def __init__(self, config:Configuration):
        super().__init__()
        self.config = config
        self.loss = create_loss(config)
        self.f1 = F1Score(threshold=C.THRESHOLD)

        # prepare pix2patch transform
        if config.model_out == 'patches' and config.loss_in == 'pixels':
            RuntimeError(f'Invalid configuration: model_out=patches, loss_in=pixels.')
        self.pix2patch = U.Pix2Patch(C.PATCH_SIZE)


    def configure_optimizers(self):
        optimizer = create_optimizer(self, self.config)
        return optimizer


    def forward(self, batch:torch.Tensor):
        n_samples, n_channels, height, width = batch.shape
        raise NotImplementedError("Must be implemented by subclass.") 


    def training_step(self, batch):
        x, targets = batch

        # forward through model to obtain probabilities
        probas = self(x)
        
        # targets always come as pixel maps
        targets_pixel, targets_patch = targets, self.pixel2patch(targets)
        # model output might come as pixels or as patches
        if self.config.model_out == 'pixel': 
            probas_pixel, probas_patch = probas, self.pixel2patch(probas)
        else: 
            probas_patch = probas # otherwise patches
        
        out = {}
        # compute loss either from pixel or patch with soft targets
        if self.config.loss_in == 'pixel':
            out['loss'] = self.loss(probas_pixel, targets_pixel)
        else:
            out['loss'] = self.loss(probas_patch, targets_patch)

        # add metrics for pixel and patch with hard targets
        if self.config.model_out == 'pixel':
            out['f1_pixel'] = self.f1(probas_pixel, targets_pixel.round())
        out['f1_patch'] = self.f1(probas_patch, targets_patch.round())
        
        return out

    #def validation_step(self, batch):
    #    out = self.training_step(batch)
    #    return {(f'val_{k}', v) for (k,v) in out.items()}

    def predict_step(self, batch):
        probas = self(batch)
        n_samples, height, width = probas.shape

        if self.config.model_out == 'pixel': 
            probas = probas, self.pixel2patch(probas)

        return probas
