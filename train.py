import os
import time

import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
from addons import SegmapVisualizer

import utils as U
from configuration import CONSTANTS as C, create_augmentation
from configuration import Configuration, create_model
from data import SatelliteData
from eval import eval


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
    config.jobid = os.environ.get('JOBID')

    # Create a logger and checkpoint file for the best model.
    logger = pl_loggers.TensorBoardLogger(save_dir=C.RESULTS_DIR, name=log_id, version='tensorboard')
    wandb = pl_loggers.WandbLogger(save_dir=C.RESULTS_DIR, config=config, project='CIL', entity='geesesquad')

    ckpt_last_cb = pl_callbacks.ModelCheckpoint(dirpath=log_dir, monitor=None,
                            filename='epoch={epoch}-step={step}-last', auto_insert_metric_name=False)
    ckpt_loss_cb = pl_callbacks.ModelCheckpoint(dirpath=log_dir, monitor='valid/loss', mode='min',
                            filename='epoch={epoch}-step={step}-val_loss={valid/loss:.3f}', auto_insert_metric_name=False)
    ckpt_f1w_patch_cb = pl_callbacks.ModelCheckpoint(dirpath=log_dir, monitor='valid/patch/f1w', mode='max',
                            filename='epoch={epoch}-step={step}-val_f1w={valid/patch/f1w:.3f}', auto_insert_metric_name=False)
    ckpt_accw_patch_cb = pl_callbacks.ModelCheckpoint(dirpath=log_dir, monitor='valid/patch/accw', mode='max',
                            filename='epoch={epoch}-step={step}-val_accw={valid/patch/accw:.3f}', auto_insert_metric_name=False)

    smap_cb = SegmapVisualizer()

    # Prepare Trainer
    trainer = pl.Trainer(
        # training dynamics
        max_epochs=config.n_epochs,
        callbacks=[ckpt_last_cb, ckpt_loss_cb, ckpt_f1w_patch_cb, ckpt_accw_patch_cb, smap_cb],

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

    # Create model
    model = create_model(config)
    print('Model created with {} trainable parameters'.format(U.count_parameters(model)))
    #wandb.watch(model=model, log='all')
    #print(model)


    # Prepare datasets and transforms.
    # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation
    aug = create_augmentation(config)
    train_set = SatelliteData('gmaps', config, transform=aug)
    valid_set = SatelliteData('training', config)
    test_set = SatelliteData('test', config, train=False)
    print(f'Init {type(train_set).__name__}: size(train)={len(train_set)}, size(valid)={len(valid_set)}')


    # Prepare datasets and dataloaders
    dl_args = dict(num_workers=config.n_workers, pin_memory=False if config.force_cpu else True ) 
    train_dl = DataLoader(train_set, config.bs_train, shuffle=True, **dl_args)
    valid_dl = DataLoader(valid_set, config.bs_eval, **dl_args)
    test_dl = DataLoader(test_set, config.bs_eval, **dl_args)
    
    # Train model.
    trainer.fit(model, train_dl, valid_dl)

    # Evaluate model and save submission
    path = os.path.splitext(ckpt_last_cb.best_model_path)[0] + '.csv'
    eval(trainer, model, valid_dl, test_dl, path)


if __name__ == '__main__':

    config = Configuration.parse_cmd()
    print(f'Starting experiment with configuration: \n {config}', flush=True)
    main(config)
