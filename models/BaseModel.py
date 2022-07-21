from typing import Dict

import pytorch_lightning as pl
import torch
import utils as U
from configuration import CONSTANTS as C
from configuration import Configuration, create_loss, create_optimizer
from torch import nn
from torchmetrics import Accuracy, F1Score


class BaseModel(pl.LightningModule):
    """A base class for neural networks that defines an interface and implements automatic
    handling of patch-wise and pixel-wise loss functions and metrics. By default, a model 
    is expected to ouput pixel-wise and patch-wise, depending on config.model_out. Please
    raise an error, if your model does not support one of the methods."""

    def __init__(self, config:Configuration):
        super().__init__()
        if config.model_out == 'patches' and config.loss_in == 'pixels':
            RuntimeError(f'Invalid configuration: model_out=patches, loss_in=pixels.')

        self.config = config
        self.loss = create_loss(config)

        ## Metrics
        self.metrics_patch = nn.ModuleDict()
        self.metrics_patch['f1_patch'] = F1Score(num_classes=1, threshold=C.THRESHOLD)
        self.metrics_patch['f1w_patch'] = F1Score(num_classes=1, average='weighted', threshold=C.THRESHOLD)
        self.metrics_patch['acc_patch'] = Accuracy(num_classes=1, threshold=C.THRESHOLD)
        self.metrics_patch['accw_patch'] = Accuracy(num_classes=1, average='weighted', threshold=C.THRESHOLD)
        
        self.metrics_pixel = nn.ModuleDict()
        if config.model_out == 'pixels':
            self.metrics_pixel['f1_pixel'] = F1Score(num_classes=1, threshold=C.THRESHOLD)
            self.metrics_pixel['f1w_pixel'] = F1Score(num_classes=1, average='weighted', threshold=C.THRESHOLD)
            self.metrics_pixel['acc_pixel'] = Accuracy(num_classes=1, threshold=C.THRESHOLD)
            self.metrics_pixel['accw_pixel'] = Accuracy(num_classes=1, average='weighted', threshold=C.THRESHOLD)
        
        # automatic pixel to patch transform by averaging 
        self.pix2patch = U.Pix2Patch(C.PATCH_SIZE)

        # prepare dimensions:
        if self.config.model_out == 'patches':
            self.out_size = int(C.IMG_SIZE / C.PATCH_SIZE)
        elif self.config.model_out == 'pixels':
            self.out_size = C.IMG_SIZE

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
        targets = batch['mask']

        # sanity checks
        # print(f'images: {images.shape}, {images.dtype}')
        # print(f'targets: {targets.shape}, {images.dtype}, {targets.max()}')

        # forward through model to obtain probabilities
        probas = self(images)

        # targets always come as pixel maps
        targets_pixel = targets.flatten()
        targets_patch = self.pix2patch(targets).flatten()

        # model output might come as pixels or as patches
        if self.config.model_out == 'pixels': 
            probas_pixel = probas.flatten()
            probas_patch = self.pix2patch(probas).flatten()

        else: # otherwise patches
            probas_patch = probas.flatten()
            

        # sanity checks
        # if self.config.model_out == 'pixels': 
        #     print(f'probas_pixel: {probas_pixel.shape}, targets_pixels: {targets_pixel.shape}')
        # print(f'probas_patch: {probas_patch.shape}, targets_patch: {targets_patch.shape}')

        out = {}
        # compute loss either from pixel or patch with soft targets
        if self.config.loss_in == 'pixels':
            out['loss'] = self.loss(probas_pixel, targets_pixel)
        else:
            out['loss'] = self.loss(probas_patch, targets_patch)

        # add metrics for pixel and patch with hard targets
        for name, metric in self.metrics_pixel: # is empty for pixelwise predictions
            out[name] = metric((probas_pixel, targets_pixel.round().int()))

        for name, metric in self.metrics_patch:
            out[name] = metric(probas_patch, (targets_patch > C.THRESHOLD).int())

        
        return out

    def training_step(self, batch:dict, batch_idx):
        out = self.step(batch, batch_idx)
        self.log_dict(dict([(f'train/{k}', v) for (k,v) in out.items()]))
        return out


    def on_validation_epoch_start(self) -> None:
        for name, metric in self.metrics_patch:
            metric.reset()
        for name, metric in self.metrics_pixel:
            metric.reset()
         
    def validation_step(self, batch:dict, batch_idx):
        out = self.step(batch, batch_idx)
        self.log('valid/loss', out['loss']) # log only loss, f1 is accumulated over all batches
        return out

    def validation_epoch_end(self, outputs) -> None:
        for name, metric in self.metrics_patch:
            self.log(f'valid/{name}', metric.compute())

        for name, metric in self.metrics_pixel:
            self.log(f'valid/{name}', metric.compute())


    def predict_step(self, batch:dict, batch_idx):
        images = batch['image']
        i_inds = batch['idx'] # image indices

        # get probabilities
        probas = self(images)
        if self.config.model_out == 'pixels': 
            probas = self.pix2patch(probas)

        # get predictions
        preds = (probas > C.THRESHOLD).int().squeeze() # remove channels

        # get submission table
        rows = []
        for k in range(preds.shape[0]):
            for i in range(preds.shape[1]):
                for j in range(preds.shape[2]):
                    row = [i_inds[k], j*C.PATCH_SIZE, i*C.PATCH_SIZE, preds[k, i, j]]
                    rows.append(torch.tensor(row).unsqueeze(0))

        return torch.cat(rows, dim=0)
