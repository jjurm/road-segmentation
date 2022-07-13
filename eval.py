import os
import argparse

import torch
from configuration import CONSTANTS as C
from configuration import Configuration, create_model
import pytorch_lightning as pl

from data import SatelliteData
from torch.utils.data import DataLoader


def main(config:Configuration, logdir:str):
    # This is just a very first draft and still under construction...
    # Needs to be checked and debugged
    # TODO: 
    # - Set seed from config?
    # - Use transforms?
    # - In train.py use methods defined here (for reproduceability)

    model = create_model(config)
    model.load_state_dict(torch.load(os.path.join(logdir, 'model.pth')))
    
    trainer = pl.Trainer()

    valid_dl = DataLoader(valid_set, config.bs_eval, num_workers=config.data_workers)
    test_dl = DataLoader(test_set, config.bs_eval, num_workers=config.data_workers)
    

    valid_set = SatelliteData('training', config)
    test_set = SatelliteData('test', config, train=False)
        # Validate model.

    trainer.validate(model, valid_dl)

    # Generate and save submission
    submission = trainer.predict(model, test_dl)
    submission = np.concatenate(submission) # concat batches
    U.to_csv(submission, os.path.join(log_dir, 'submission.csv'))

    pass


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

    print(f'Evaluating experiment evaluation with configuration: \n {config}', flush=True)
    #main(config, args.log_dir)

