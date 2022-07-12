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
    pl.seed_everything(config.seed)

    # Create or an experiment ID and a folder where to store logs and config.
    log_dir, log_id = U.create_log_dir(config)
    U.export_cmd(os.path.join(log_dir, 'cmd.sh'))
    U.export_code(os.path.join(log_dir, 'code.zip'))
    config.to_json(os.path.join(log_dir, 'config.json'))
    config.log_id = log_id # so that it is also accessible from wandb

    # Create model
    model = create_model(config)
    print('Model created with {} trainable parameters'.format(U.count_parameters(model)))
    print(model)

    # Prepare datasets and transforms.
    # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation
    train_set = SatelliteData('gmaps', config)
    valid_set = SatelliteData('training', config)
    test_set = SatelliteData('test', config, train=False)
    print(f'Init {type(train_set).__name__}: size(train)={len(train_set)}, size(valid)={len(valid_set)}')


    # Create a logger and checkpoint file for the best model.
    logger = pl_loggers.TensorBoardLogger(save_dir=C.RESULTS_DIR, name=log_id, version='tensorboard')
    wandb = pl_loggers.WandbLogger(save_dir=C.RESULTS_DIR, config=config, project='CIL')
    wandb.watch(model=model, log='all')
    checkpoint_cb = pl_callbacks.ModelCheckpoint(dirpath=log_dir, filename='model.pth', monitor='valid/loss')

    # Pepare Trainer
    trainer = pl.Trainer(
        # training dynamics
        max_epochs=config.n_epochs,
        callbacks=[checkpoint_cb],

        # logging
        logger=[logger, wandb],
        log_every_n_steps=config.log_every,

        # acceleration
        accelerator='cpu' if config.force_cpu else 'gpu',
        devices=None if config.force_cpu else 1,
        auto_select_gpus=True,

        # debugging
        #limit_train_batches=2,
        #limit_val_batches=2,
        )


    # Prepare datasets and dataloaders
    dl_args = dict(num_workers=config.n_workers, pin_memory=False if config.force_cpu else True ) 
    train_dl = DataLoader(train_set, config.bs_train, **dl_args)
    valid_dl = DataLoader(valid_set, config.bs_eval, **dl_args)
    test_dl = DataLoader(test_set, config.bs_eval, **dl_args)
    
    # Train model.
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
