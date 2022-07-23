import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score

import utils as U
import wandb
from addons import SegmapVisualizer, TrainMetricLogger, ValidMetricLogger
from configuration import CONSTANTS as C
from configuration import Configuration, create_model
from data import SatelliteData
from models.BaseModel import BaseModel


def eval(trainer:pl.Trainer, 
            model:BaseModel, 
            valid_dl:DataLoader, 
            test_dl:DataLoader,
            path:str):

    # Validate model.
    trainer.validate(model, valid_dl) #TODO: does this auto_select GPU?

    # Generate and save submission
    submission = trainer.predict(model, test_dl)
    submission = torch.cat(submission, dim=0) # concat batches
    
    U.to_csv(submission.cpu(), path)

def main(config:Configuration):

    log_callbacks = [
        TrainMetricLogger(model_out=config.model_out, t_metrics={'f1':F1Score, 'acc':Accuracy}, weighted=True),
        ValidMetricLogger(model_out=config.model_out, t_metrics={'f1':F1Score, 'acc':Accuracy}, weighted=True),
    ]
    
    # Prepare Trainer
    trainer = pl.Trainer(
        logger=log_callbacks,

        # acceleration
        accelerator='cpu' if config.force_cpu else 'gpu',
        devices=None if config.force_cpu else 1,
        auto_select_gpus=True,

        # debugging
        #limit_train_batches=2,
        #limit_val_batches=2,
        )

    # TODO: use augmentations?
    valid_set = SatelliteData('training', config)
    test_set = SatelliteData('test', config, train=False)
    print(f'Init {type(valid_set).__name__}: size(valid)={len(valid_set)}, size(test)={len(test_set)}')

    # Prepare datasets and dataloaders
    dl_args = dict(num_workers=config.n_workers, pin_memory=False if config.force_cpu else True ) 
    valid_dl = DataLoader(valid_set, config.bs_eval, **dl_args)
    test_dl = DataLoader(test_set, config.bs_eval, **dl_args)
    
    model = create_model(config)
    model = model.load_from_checkpoint(
                checkpoint_path=config.ckpt, 
                device=torch.device('cpu') if config.force_cpu else None, 
                config=config)
    
    # Evaluate model and save submission
    path = os.path.splitext(config.ckpt)[0] + '.csv'
    eval(trainer, model, valid_dl, test_dl, path)


if __name__ == '__main__':

    # Initial parser.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('ckpt', type=str,
                        help='Load state_dict and configuration from checkpoint file.')
    
    # Argument parser to overwrite evaluation-specific args.
    parser = argparse.ArgumentParser(parents=[pre_parser],
                formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--n_workers', type=int, default=4, 
                            help='Number of parallel threads for data loading.')
    parser.add_argument('--force_cpu', action='store_true',
                            help='Force training on CPU instead of GPU.')
    parser.add_argument('--bs_eval', type=int, default=16, 
                            help='Batch size for valid/test set.')

    # 1. Get default configurations
    config = Configuration.get_default()

    # 2. Overwrite with defaults with checkpoint config
    args = parser.parse_args()
    ckpt = torch.load(args.ckpt, torch.device('cpu'))
    config.update(ckpt['hyper_parameters']['config'])

    # 2. Overwrite checkpoint config with remaining cmd args
    parser.parse_args(namespace=config)

    print(f'Evaluating experiment with configuration: \n {config}', flush=True)
    main(config)

