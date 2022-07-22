from copy import deepcopy
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

        self.config = config
        self.loss = create_loss(config)

        if config.model_out == 'patch' and config.loss_in == 'pixel':
            RuntimeError(f'Invalid configuration: model_out=patch, loss_in=pixel.')

        # prepare dimensions:
        if self.config.model_out == 'patch':
            self.out_size = int(C.IMG_SIZE / C.PATCH_SIZE)

        elif self.config.model_out == 'pixel':
            self.out_size = C.IMG_SIZE

        # automatic pixel to patch transform by averaging 
        self.pix2patch = U.Pix2Patch(C.PATCH_SIZE)

        # output modes of model: for pixelwise model, also the patchwise outputs are tracked
        self.outmodes = ['patch', 'pixel'] if (config.model_out=='pixel') else ['patch']

        ## Metrics
        # sadly can't reuse metrics: https://torchmetrics.readthedocs.io/en/latest/pages/lightning.html
        self.train_metrics = nn.ModuleDict()
        self.valid_metrics = nn.ModuleDict()
        for phase_metrics in [self.train_metrics, self.valid_metrics]:
            for outmode in self.outmodes:
                phase_metrics[outmode] = nn.ModuleDict()
                phase_metrics[outmode]['f1'] = F1Score(num_classes=None, multiclass=None)   #binary
                phase_metrics[outmode]['acc'] = Accuracy(num_classes=None, multiclass=None) #binary
                phase_metrics[outmode]['f1w'] = F1Score(num_classes=2, multiclass=True, average='weighted')   #multiclass weighted
                phase_metrics[outmode]['accw'] = Accuracy(num_classes=2, multiclass=True, average='weighted') #multiclass weighted

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

    def get_hardlabels(self, out:Dict[str, Dict[str, torch.Tensor]], outmode:str) -> dict:
        threshold = C.THRESHOLD if (outmode=='patch') else 0.5
        preds = (out[outmode]['probas'] > threshold).float() # convert to hard labels
        targs = (out[outmode]['targets'] > threshold).int() # convert to hard labels
        return preds, targs


    def training_step(self, batch:dict, batch_idx):
        out = self.step(batch, batch_idx)

        # store loss to a logging dictionary
        logdict = {'train/loss' : out['loss']}

        # compute pixel- and patchwise predicitons/targes as hard labels
        for outmode in self.train_metrics.keys():  
            preds, targs = self.get_hardlabels(out, outmode)
            preds, targs = preds.flatten(), targs.flatten() # need to flatten for metrics

            # compute every metric, than add to logdict
            for name in self.train_metrics[outmode].keys():     
                metric_value = self.train_metrics[outmode][name](preds, targs)      # compute metric
                logdict[f'train/{outmode}/{name}'] = metric_value               # add to logdict
        
        self.log_dict(logdict)
        return out
    
    def train_epoch_end(self) -> None:
        # reset every metric
        for outmode in self.train_metrics.keys():
            for name in self.train_metrics[outmode].keys():
                self.train_metrics[outmode][name].reset()


    def validation_step(self, batch:dict, batch_idx):
        out = self.step(batch, batch_idx)
        
        # store loss to a logging dictionary
        logdict = {'valid/loss' : out['loss']}

        # compute pixel- and patchwise predicitons/targes as hard labels
        for outmode in self.valid_metrics.keys():  
            preds, targs = self.get_hardlabels(out, outmode)
            preds, targs = preds.flatten(), targs.flatten() # need to flatten for metrics
            
            # update every metric, computation will happen on validation_epoch_end
            for name in self.valid_metrics[outmode].keys():     
                self.valid_metrics[outmode][name].update(preds, targs)      # update metric
        
        self.log_dict(logdict)
        return out

    def validation_epoch_end(self, outputs) -> None:
        logdict = {}

        # for compute, log and reset every validation metric
        for outmode in self.valid_metrics.keys():  
            for name in self.valid_metrics[outmode].keys():
                metric_value = self.valid_metrics[outmode][name].compute()     # compute metric
                logdict[f'valid/{outmode}/{name}'] = metric_value               # add to logdict
                
                self.valid_metrics[outmode][name].reset()               
        
        self.log_dict(logdict)


    def predict_step(self, batch:dict, batch_idx):
        images = batch['image']
        i_inds = batch['idx'] # image indices

        # get probabilities
        probas = self(images)
        if self.config.model_out == 'pixel': 
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
