import os
import time
import pandas as pd
import pytorch_lightning as pl

import utils as U
from configuration import CONSTANTS as C
from configuration import Configuration, create_model

from data import SatelliteData
from torch.utils.data import DataLoader

def main(config:Configuration):

    # Fix random seed
    if config.seed is None:
        config.seed = int(time.time())

    # Create or an experiment ID and a folder where to store logs and config.
    log_dir = U.create_model_dir(config)
    U.export_cmd(os.path.join(log_dir, 'config.json'))
    U.export_code(os.path.join(log_dir, 'config.json'))
    config.to_json(os.path.join(log_dir, 'config.json'))

    # Create a checkpoint file for the best model.
    checkpoint_file = os.path.join(log_dir, 'model.pth')
    print('Saving checkpoints to {}'.format(checkpoint_file))

    # Create model on correct device
    model = create_model(config).to(C.DEVICE)
    print('Model created with {} trainable parameters'.format(U.count_parameters(model)))
    print(model)

    # Prepare datasets and Dataloaders
    train_set = SatelliteData('gmaps', config)
    valid_set = SatelliteData('train', config)
    test_set = SatelliteData('test', config)
    
    train_dl = DataLoader(train_set, config.bs_train, num_workers=config.data_workers)
    valid_dl = DataLoader(valid_set, config.bs_eval, num_workers=config.data_workers)
    test_dl = DataLoader(test_set, config.bs_eval, num_workers=config.data_workers)
    
    # Train model.
    trainer = pl.Trainer(max_epochs=config.n_epochs)
    trainer.fit(model, train_dl, valid_dl)

    # Validate model.
    trainer.validate(model, valid_dl)

    # Generate predictions and store submission
    preds = trainer.predict(model, test_dl)
    preds = pd.DataFrame(preds)

    preds.to_csv(os.path.join(log_dir, 'submission.scv'))
    print(preds.head(10))


    return


if __name__ == '__main__':

    config = Configuration.parse_cmd()
    print(f'Starting experiment with configuration: \n {config}', flush=True)
    main(config)
