import os
from typing import Dict, Type

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning import loggers as pl_loggers
from torch import nn
from torchmetrics import Metric

import utils as U
from configuration import CONSTANTS as C
from models.BaseModel import BaseModel


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


    def reset_metrics(self, device):
        self.metrics.to(device)
        for outmode in self.metrics.keys():  
            for name in self.metrics[outmode].keys():
                self.metrics[outmode][name].reset()  


class TrainMetricLogger(MetricLoggerBase):
    def on_train_epoch_start(self, trainer: pl.Trainer, model: BaseModel) -> None:
        self.reset_metrics(model.device)

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
    def on_validation_epoch_start(self, trainer: pl.Trainer, model: BaseModel) -> None:
        self.reset_metrics(model.device)

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

        self.upsample = torch.nn.Upsample(scale_factor=16)


    def on_fit_start(self, trainer: "pl.Trainer", model: BaseModel) -> None:
        self.imgs_rgb = self.imgs_rgb.to(model.device)
        self.imgs_gt = self.imgs_gt.to(model.device)
    
    def color_fp_fn(self, probas, targs, outmode, model:BaseModel):
        preds = model.apply_threshold(probas, outmode).bool()
        targs = model.apply_threshold(targs, outmode).bool()

        tp = (preds * targs) # true positive:  prediction and target are 1
        fp = (preds > targs) # false positive: predicition is 1, target 0
        fn = (preds < targs) # false negative: prediction is 0, target 1


        # true positive => white
        cmap = torch.zeros_like(probas)
        cmap[tp] = 1

        # false positive => green
        fp[:,0,:,:] = 0     # cancel red
        fp[:,2,:,:] = 0     # cancel blue
        cmap[fp] = 1

        # false negative => red
        fn[:,1,:,:] = 0     # cancel green
        fn[:,2,:,:] = 0     # cancel blue
        cmap[fn] = 1

        return cmap

    def on_validation_epoch_end(self, trainer: "pl.Trainer", model: BaseModel) -> None:

        # get predictions
        B,_,H,W = self.imgs_rgb.shape
        pred = model.sigmoid(model(self.imgs_rgb))
        imgs_pred_pix = pred.repeat(1,3,1,1)

        # get patches from pixels and upsample to original size
        imgs_gt_patches = model.pix2patch(self.imgs_gt)
        
        # log validation images depending on the model output
        if model.config.model_out == 'pixel':
            imgs_pred_patches = model.pix2patch(imgs_pred_pix)
            imgs_fp_fn_patches = self.color_fp_fn(imgs_pred_patches, imgs_gt_patches, 'patch', model)
            imgs_fp_fn_pix = self.color_fp_fn(imgs_pred_pix, self.imgs_gt, 'pixel', model)
            visualization_plan = [
                (self.imgs_rgb, self.rgb_tags),
                (self.imgs_gt, 'GT (pixel)'),
                (imgs_pred_pix, 'Pred (pixel)'),
                (imgs_fp_fn_pix, 'Error (pixel): FP green, FN red'),
                (self.upsample(imgs_gt_patches), 'GT (patch)'),
                (self.upsample(imgs_pred_patches), 'Pred (patch)'),
                (self.upsample(imgs_fp_fn_patches), 'Error (patch): FP green, FN red'),
            ]
        else:
            imgs_pred_patches = imgs_pred_pix
            imgs_fp_fn_patches = self.color_fp_fn(imgs_pred_patches, imgs_gt_patches, 'patch', model)
            visualization_plan = [
                (self.imgs_rgb, self.rgb_tags),
                (self.imgs_gt, 'GT (pixel)'),
                (self.upsample(imgs_gt_patches), 'GT (patch'),
                (self.upsample(imgs_pred_patches), 'Pred (patch)'),
                (self.upsample(imgs_fp_fn_patches), 'Error (patch): FP green, FN red'),
            ]
        
        # merge all pictures in 1 grid-like image
        vis = U.compose(visualization_plan)

        for wb_logger in model.loggers:
            if not isinstance(wb_logger, pl_loggers.WandbLogger):
                continue
            wb_logger.log_image('val_pred', images=[vis], caption=["Validation results"])
