import pytorch_lightning as pl
import torch
import utils as U
from configuration import CONSTANTS as C
from configuration import Configuration, create_loss, create_optimizer
from torchmetrics import F1Score
from PIL import Image
import os
import wandb
import torchvision.transforms as transforms
import numpy as np


class BaseModel(pl.LightningModule):
    """A base class for neural networks that defines an interface and implements automatic
    handling of patch-wise and pixel-wise loss functions and metrics. By default, a model 
    is expected to ouput pixel-wise and patch-wise, depending on config.model_out. Please
    raise an error, if your model does not support one of the methods."""

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


    def on_validation_epoch_start(self) -> None:
        self.f1_pixel.reset()
        self.f1_patch.reset()

    def validation_step(self, batch:dict, batch_idx):
        out = self.step(batch, batch_idx)
        self.log('valid/loss', out['loss']) # log only loss, f1 is accumulated over all batches
        return out

    def validation_epoch_end(self, outputs) -> None:
        if self.config.model_out == 'pixels':
            self.log('valid/f1_pixel', self.f1_pixel.compute())
        self.log('valid/f1_patch', self.f1_patch.compute())

        # === Qualitative feedback ===
        # -- choose images --
        val_images = ['satimage_2', 'satimage_15', 'satimage_88', 'satimage_90', 'satimage_116']

        # get path to images
        rgb_tags = [val_image + ".png" for val_image in val_images]
        paths_to_img = [os.path.join(C.DATA_DIR, 'training', 'images', rgb_tag) for rgb_tag in rgb_tags]
        paths_to_gt = [os.path.join(C.DATA_DIR, 'training', 'groundtruth', rgb_tag) for rgb_tag in rgb_tags]

        # convert images to tensors of same size
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        imgs_rgb = torch.stack([transform(Image.open(path_to_img)).type(torch.float32)[:3,:,:] / 255.0
            for path_to_img in paths_to_img]).to(self.device)
        imgs_gt = torch.stack([transform(Image.open(path_to_gt)).type(torch.float32).repeat(3,1,1) / 255.0
            for path_to_gt in paths_to_gt]).to(self.device)

        # get predictions
        B,_,H,W = imgs_rgb.shape
        pred = self(imgs_rgb)
        imgs_pred_pix = pred.unsqueeze(1).repeat(1,3,1,1)

        # get patches from pixels and upsample to original size
        imgs_gt_patches = self.pix2patch(imgs_gt[:,0,:,:]).unsqueeze(1).repeat(1,3,1,1)
        upsample = torch.nn.Upsample(scale_factor=16)
        imgs_gt_patches = upsample(imgs_gt_patches)
        
        # log validation images depending on the model output
        if self.config.model_out == 'pixels':
            imgs_pred_patches = self.pix2patch(imgs_pred_pix[:,0,:,:]).unsqueeze(1).repeat(1,3,1,1)
            imgs_pred_patches = upsample(imgs_pred_patches)
            visualization_plan = [
                (imgs_rgb, rgb_tags),
                (imgs_gt, 'GT (pixels'),
                (imgs_pred_pix, 'Pred (pixels)'),
                (imgs_gt_patches, 'GT (patches)'),
                (imgs_pred_patches, 'Pred (patches)'),
            ]
        else:
            imgs_pred_patches = imgs_pred_pix
            imgs_pred_patches = upsample(imgs_pred_patches)
            visualization_plan = [
                (imgs_rgb, rgb_tags),
                (imgs_gt, 'GT (pixels'),
                (imgs_gt_patches, 'GT (patches'),
                (imgs_pred_patches, 'Pred (patches)'),
            ]
        
        # merge all pictures in 1 grid-like image
        vis = U.compose(visualization_plan)

        wandb.log({
            'val_pred': [wandb.Image(vis.cpu(), caption="Validation results")]
        })


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
        return submission.reshape(-1, 4).cpu().numpy()
