import os
import pytorch_lightning as pl
import torch
import wandb

from configuration import CONSTANTS as C
import torchvision.transforms as transforms
from PIL import Image

import utils as U

class SegmapVisualizer(pl.Callback):
    def __init__(self, val_images=None) -> None:
        super().__init__()

    # === Qualitative feedback ===
        # -- choose images --
        if val_images is None:
            val_images = ['satimage_2', 'satimage_15', 'satimage_88', 'satimage_90', 'satimage_116']

        # get path to images
        self.rgb_tags = [val_image + ".png" for val_image in val_images]
        paths_to_img = [os.path.join(C.DATA_DIR, 'training', 'images', rgb_tag) for rgb_tag in self.rgb_tags]
        paths_to_gt = [os.path.join(C.DATA_DIR, 'training', 'groundtruth', rgb_tag) for rgb_tag in self.rgb_tags]

        # convert images to tensors of same size
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        self.imgs_rgb = torch.stack([transform(Image.open(path_to_img)).type(torch.float32)[:3,:,:] / 255.0
            for path_to_img in paths_to_img])
        self.imgs_gt = torch.stack([transform(Image.open(path_to_gt)).type(torch.float32).repeat(3,1,1) / 255.0
            for path_to_gt in paths_to_gt])

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.imgs_rgb = self.imgs_rgb.to(pl_module.device)
        self.imgs_gt = self.imgs_gt.to(pl_module.device)
    
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        # get predictions
        B,_,H,W = self.imgs_rgb.shape
        pred = pl_module(self.imgs_rgb)
        imgs_pred_pix = pred.unsqueeze(1).repeat(1,3,1,1)

        # get patches from pixels and upsample to original size
        imgs_gt_patches = pl_module.pix2patch(self.imgs_gt[:,0,:,:]).unsqueeze(1).repeat(1,3,1,1)
        upsample = torch.nn.Upsample(scale_factor=16)
        imgs_gt_patches = upsample(imgs_gt_patches)
        
        # log validation images depending on the model output
        if pl_module.config.model_out == 'pixels':
            imgs_pred_patches = pl_module.pix2patch(imgs_pred_pix[:,0,:,:]).unsqueeze(1).repeat(1,3,1,1)
            imgs_pred_patches = upsample(imgs_pred_patches)
            visualization_plan = [
                (self.imgs_rgb, self.rgb_tags),
                (self.imgs_gt, 'GT (pixels'),
                (imgs_pred_pix, 'Pred (pixels)'),
                (imgs_gt_patches, 'GT (patches)'),
                (imgs_pred_patches, 'Pred (patches)'),
            ]
        else:
            imgs_pred_patches = imgs_pred_pix
            imgs_pred_patches = upsample(imgs_pred_patches)
            visualization_plan = [
                (self.imgs_rgb, self.rgb_tags),
                (self.imgs_gt, 'GT (pixels'),
                (imgs_gt_patches, 'GT (patches'),
                (imgs_pred_patches, 'Pred (patches)'),
            ]
        
        # merge all pictures in 1 grid-like image
        vis = U.compose(visualization_plan)

        wandb.log({
            'val_pred': [wandb.Image(vis.cpu(), caption="Validation results")]
        })