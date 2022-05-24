import os
from datetime import datetime

import torch

from configuration import CONSTANTS as C
from configuration import Configuration


def main(config):
    pass



if __name__ == '__main__':

    config = Configuration.parse_cmd()
    print(f'Starting experiment with configuration: \n {config}', flush=True)

    # Make experiment specific directory
    id = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    C.EXPERIMENT_DIR = os.path.join(C.EXPERIMENT_DIR, id)
    os.makedirs(C.EXPERIMENT_DIR, exist_ok = True) 

    # Store configuration to experiment directory
    config.to_json(os.path.join(C.EXPERIMENT_DIR, 'config.json'))

    main(config)
