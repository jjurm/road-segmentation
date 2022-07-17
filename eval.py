import os
import argparse

import numpy as np
import torch
import wandb

from configuration import CONSTANTS as C
from configuration import Configuration, create_model
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from models.BaseModel import BaseModel

import utils as U
from data import SatelliteData
from torch.utils.data import DataLoader

def eval(trainer:pl.Trainer, 
            model:BaseModel, 
            valid_dl:DataLoader, 
            test_dl:DataLoader):

    # Validate model.
    trainer.validate(model, valid_dl) #TODO: does this auto_select GPU?

    # Generate and save submission
    submission = trainer.predict(model, test_dl)
    submission = np.concatenate(submission) # concat batches
    
    path = os.path.join(config.log_dir, 'submission.csv')
    U.to_csv(submission, path)


def main(config:Configuration):
    # Prepare Trainer

    trainer = pl.Trainer(
        logger=None,

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

    # Load model to cpu
    checkpoint_path = os.path.join(config.log_dir, 'model.ckpt')
    model = create_model(config)
    model = model.load_from_checkpoint(
                checkpoint_path=checkpoint_path, 
                device=torch.device('cpu'), 
                config=config)
    
    # Evaluate model and save submission
    eval(trainer, model, valid_dl, test_dl)


if __name__ == '__main__':

    # Initial parser.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('log_dir', type=str,
                        help='Load checkpoint and configuration from log directory.')
    
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

    # 2. Overwrite with defaults with JSON config
    args = parser.parse_args()
    if args.log_dir is not None:
        json_path = os.path.join(args.log_dir, 'config.json') 
        config = Configuration.from_json(json_path, config)

    # 2. Overwrite json with remaining cmd args
    parser.parse_args(namespace=config)

    print(f'Evaluating experiment with configuration: \n {config}', flush=True)
    main(config)

