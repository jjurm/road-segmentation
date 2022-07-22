import os
from typing import Dict, Type
import pytorch_lightning as pl
from sklearn import metrics
import torch
import wandb

from configuration import CONSTANTS as C
import torchvision.transforms as transforms
from PIL import Image
from models.BaseModel import BaseModel

import utils as U
import torch
from torch import nn
from torchmetrics import Metric

class MetricLoggerBase(pl.Callback):
    '''
    MetricLogger Callback, which tries to adhere to the torchmetrics documentation for pytorch-lightning:
    https://torchmetrics.readthedocs.io/en/latest/pages/lightning.html
    '''
    def __init__(self, model_out:str, t_metrics:Dict[str, Type[Metric]], weighted=True) -> None:
        super().__init__()
                
        # output modes of model: for pixelwise model, also the patchwise outputs are tracked
        self.outmodes = ['patch', 'pixel'] if (model_out=='pixel') else ['patch']

        # It's not possible to reuse metrics.. thus we need to store all of them independently
        self.metrics = nn.ModuleDict()
        
        for outmode in self.outmodes:  # for every output mode, store a dictionary of metrics
            self.metrics[outmode] = nn.ModuleDict()

            for name, t_metric in t_metrics.items(): # add all specified metrics to dictionary
                # instanciate the metric as a binary metric
                self.metrics[outmode][name] = t_metric(num_classes=None, multiclass=None) #binary

                if weighted: # instanciate the metric as a binary metric with weighted averaging
                     self.metrics[outmode][name+'w'] = t_metric(num_classes=2, multiclass=True, average='weighted')
        return


    def reset_metrics(self):
        for outmode in self.metrics.keys():  
            for name in self.metrics[outmode].keys():
                self.metrics[outmode][name].reset()  


class TrainMetricLogger(MetricLoggerBase):
    def on_train_epoch_start(self, *args) -> None:
        self.reset_metrics()

    def on_train_batch_end(self, trainer: pl.Trainer, model: BaseModel, out:Dict[str, Dict[str, torch.Tensor]], *args):
        logdict = {} 
        # compute pixel- and patchwise predicitons/targes as hard labels
        for outmode in self.metrics.keys():  
            preds = model.apply_threshold(out[outmode]['probas'], outmode)
            targs = model.apply_threshold(out[outmode]['targets'], outmode)

            # need to flatten and cast for metrics
            preds, targs = preds.float().flatten(), targs.int().flatten() 

            # compute every metric, than batch_value to logdict
            for name in self.metrics[outmode].keys():     
                batch_value = self.metrics[outmode][name](preds, targs)      # compute metric
                logdict[f'train/{outmode}/{name}'] = batch_value               # add to logdict
        
        model.log_dict(logdict)


class ValidMetricLogger(MetricLoggerBase):
    def on_validation_epoch_start(self, *args) -> None:
        self.reset_metrics() 

    def on_validation_batch_end(self, trainer: pl.Trainer, model: BaseModel, out:Dict[str, Dict[str, torch.Tensor]], *args):
        # compute pixel- and patchwise predicitons/targes as hard labels
        for outmode in self.metrics.keys():  
            preds = model.apply_threshold(out[outmode]['probas'], outmode)
            targs = model.apply_threshold(out[outmode]['targets'], outmode)

            # need to flatten and cast for metrics
            preds, targs = preds.float().flatten(), targs.int().flatten() 
            
            # update every metric, computation will happen on validation_epoch_end
            for name in self.metrics[outmode].keys():     
                self.metrics[outmode][name].update(preds, targs)      # update metric
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, model: BaseModel):
        logdict = {}
        # compute, log and reset every validation metric
        for outmode in self.metrics.keys():  
            for name in self.metrics[outmode].keys():
                metric_value = self.metrics[outmode][name].compute()     # compute metric
                logdict[f'valid/{outmode}/{name}'] = metric_value               # add to logdict
        
        model.log_dict(logdict)



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
        imgs_pred_pix = pred.repeat(1,3,1,1)

        # get patches from pixels and upsample to original size
        imgs_gt_patches = pl_module.pix2patch(self.imgs_gt[:,0,:,:]).repeat(1,3,1,1)
        upsample = torch.nn.Upsample(scale_factor=16)
        imgs_gt_patches = upsample(imgs_gt_patches)
        
        # log validation images depending on the model output
        if pl_module.config.model_out == 'pixel':
            imgs_pred_patches = pl_module.pix2patch(imgs_pred_pix[:,0,:,:]).repeat(1,3,1,1)
            imgs_pred_patches = upsample(imgs_pred_patches)
            visualization_plan = [
                (self.imgs_rgb, self.rgb_tags),
                (self.imgs_gt, 'GT (pixel'),
                (imgs_pred_pix, 'Pred (pixel)'),
                (imgs_gt_patches, 'GT (patch)'),
                (imgs_pred_patches, 'Pred (patch)'),
            ]
        else:
            imgs_pred_patches = imgs_pred_pix
            imgs_pred_patches = upsample(imgs_pred_patches)
            visualization_plan = [
                (self.imgs_rgb, self.rgb_tags),
                (self.imgs_gt, 'GT (pixel'),
                (imgs_gt_patches, 'GT (patch'),
                (imgs_pred_patches, 'Pred (patch)'),
            ]
        
        # merge all pictures in 1 grid-like image
        vis = U.compose(visualization_plan)

        wandb.log({
            'val_pred': [wandb.Image(vis.cpu(), caption="Validation results")]
        })