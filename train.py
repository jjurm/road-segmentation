import os
import time

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

import utils as U
from configuration import CONSTANTS as C
from configuration import Configuration, create_model
from data import SatelliteData


def main(config:Configuration):

    # Fix random seed
    if config.seed is None:
        config.seed = int(time.time())
    # TODO: use seed in Trainer and Albumentations?

    # Create or an experiment ID and a folder where to store logs and config.
    log_dir, log_id = U.create_log_dir(config)
    U.export_cmd(os.path.join(log_dir, 'cmd.sh'))
    U.export_code(os.path.join(log_dir, 'code.zip'))
    config.to_json(os.path.join(log_dir, 'config.json'))

    # Create a logger and checkpoint file for the best model.
    logger = pl_loggers.TensorBoardLogger(save_dir=C.RESULTS_DIR, name=log_id, version='tensorboard')
    checkpoint_cb = pl_callbacks.ModelCheckpoint(dirpath=log_dir, filename='model.pth', monitor='val_loss')

    # Create model on correct device
    model = create_model(config)
    print('Model created with {} trainable parameters'.format(U.count_parameters(model)))
    print(model)

    # Prepare Transforms
    # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation

    # Prepare datasets and Dataloaders
    train_set = SatelliteData('gmaps', config)
    valid_set = SatelliteData('training', config)
    test_set = SatelliteData('test', config, train=False)
    
    train_dl = DataLoader(train_set, config.bs_train, num_workers=config.data_workers)
    valid_dl = DataLoader(valid_set, config.bs_eval, num_workers=config.data_workers)
    test_dl = DataLoader(test_set, config.bs_eval, num_workers=config.data_workers)
    
    
    # Train model.
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                        accelerator='auto',
                        logger=logger,
                        callbacks=[checkpoint_cb])
    trainer.fit(model, train_dl, valid_dl)

    # Validate model.
    trainer.validate(model, valid_dl)

    # Generate and save submission
    submission = trainer.predict(model, test_dl)
    submission = np.concatenate(submission) # concat batches
    U.to_csv(submission, os.path.join(log_dir, 'submission.csv'))

    return


if __name__ == '__main__':

    config = Configuration.parse_cmd()
    print(f'Starting experiment with configuration: \n {config}', flush=True)
    main(config)
