import os
import argparse

import torch
from configuration import CONSTANTS as C
from configuration import Configuration, create_model
import pytorch_lightning as pl

from data import SatelliteData
from torch.utils.data import DataLoader


def main(config:Configuration, logdir:str):
    # This is just a very first draft...
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str,
                        help='Load configuration from JSON file.')
    args = parser.parse_args()

    config = Configuration.from_json(os.path.join(args.log_dir, 'config.json'))
    print(f'Evaluating experiment with configuration: \n {config}', flush=True)
    main(config, args.slogdir)
