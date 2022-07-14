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
        self.f1_pixel = F1Score()
        self.f1_patch = F1Score(threshold=C.THRESHOLD)

        # prepare pix2patch transform
        if config.model_out == 'patches' and config.loss_in == 'pixels':
            RuntimeError(f'Invalid configuration: model_out=patches, loss_in=pixels.')
        self.pix2patch = U.Pix2Patch(C.PATCH_SIZE)

        # prepare dimensions:
        if self.config.model_out == 'patches':
            self.out_size = int(C.IMG_SIZE / C.PATCH_SIZE)
        elif self.config.model_out == 'pixels':
            self.out_size = C.IMG_SIZE


    def configure_optimizers(self):
        optimizer = create_optimizer(self, self.config)
        return optimizer


    def forward(self, batch:torch.Tensor):
        n_samples, n_channels, in_size, in_size = batch.shape
        raise NotImplementedError("Must be implemented by subclass.")
        return batch.reshape(n_samples, self.out_size, self.out_size) # remove channel dim

    def step(self, batch:dict, batch_idx):
        images = batch['image']
        targets = batch['mask']

        # sanity checks
        # print(f'images: {images.shape}, {images.dtype}')
        # print(f'targets: {targets.shape}, {images.dtype}, {targets.max()}')

        # forward through model to obtain probabilities
        probas = self(images)

        # targets always come as pixel maps
        targets_pixel, targets_patch = targets, self.pix2patch(targets)
        # model output might come as pixels or as patches
        if self.config.model_out == 'pixels': 
            probas_pixel, probas_patch = probas, self.pix2patch(probas)
        else: 
            probas_patch = probas # otherwise patches

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
        if self.config.model_out == 'pixels':
            out['f1_pixel'] = self.f1_pixel(probas_pixel, targets_pixel.int()) # TODO: do we need to round with augmentations?
        out['f1_patch'] = self.f1_patch(probas_patch, (targets_patch > C.THRESHOLD).int())
        
        return out

    def training_step(self, batch:dict, batch_idx):
        out = self.step(batch, batch_idx)
        self.log_dict(dict([(f'train/{k}', v) for (k,v) in out.items()]))
        return out

    def validation_step(self, batch:dict, batch_idx):
        out = self.step(batch, batch_idx)
        self.log_dict(dict([(f'valid/{k}', v) for (k,v) in out.items()]))
        return out


    def predict_step(self, batch:dict, batch_idx):
        images = batch['image']
        i_inds = batch['idx'] # image indices

        # get probabilities
        probas = self(images)  
        if self.config.model_out == 'pixels': 
            probas = self.pix2patch(probas)

        # get predictions
        preds = (probas > C.THRESHOLD).int() * 255

        # generate patch indices for one sample: [size, size, 2]
        n_samples, size, _ = preds.shape
        p_inds_W = torch.arange(0, C.IMG_SIZE, C.PATCH_SIZE, device=self.device)
        p_inds_W = p_inds_W.expand(size, size)
        p_inds_H = p_inds_W.transpose(0,1)
        p_inds = torch.stack((p_inds_H, p_inds_W), -1)

        # prepare indices and preds for patch: [n_samples, size, size, ]
        p_inds = p_inds.expand(n_samples, size, size, 2) # p_inds for every batch
        i_inds = i_inds.reshape(n_samples, 1, 1, 1)
        i_inds = i_inds.expand(n_samples, size, size, 1) # i_inds for every patch
        preds = preds.unsqueeze(-1) # singelton dimension for prediction

        # reshape into submission format
        submission = torch.cat([i_inds, p_inds, preds], -1)
        return submission.reshape(-1, 4).numpy()
