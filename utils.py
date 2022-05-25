import os
import re
import sys
import zipfile
from datetime import datetime
from glob import glob

import torch
from torch.nn import functional as F

from configuration import CONSTANTS as C
from configuration import Configuration


def create_log_dir(config:Configuration):
    '''
    Create a new logging directory with an id containing timestamp and model name.
    
    Args:
        config: Configuration of current experiment.

    Returns:
        str: A directory where we can store logs. Raises an exception if the model directory already exists.    
    '''

    id = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    log_dir = os.path.join(C.RESULTS_DIR, f'{id}--{config.model}')

    # make experiments directory
    os.makedirs(C.RESULTS_DIR, exist_ok=True)

    if os.path.exists(log_dir):
        raise ValueError('Logging directory already exists {}'.format(log_dir))
    os.makedirs(log_dir)

    return log_dir


def export_cmd(fname):
    '''Stores command to a .txt file.'''
    if not fname.endswith('.txt'):
        fname += '.txt'

    cmd = ' '.join(sys.argv)
    with open(fname, 'w') as f:
        f.write(cmd)


def export_code(fname):
    '''Stores .py source files in a zip.'''
    if not fname.endswith('.zip'):
        fname += '.zip'

    code_files = glob('./*.py', recursive=True)
    zipf = zipfile.ZipFile(fname, mode='w', compression=zipfile.ZIP_DEFLATED)
    for f in code_files:
        zipf.write(f)
    zipf.close()


def count_parameters(net):
    '''Count number of trainable parameters in `net`.'''
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def get_filenames(dataset, subdir):
    '''Get datafiles from dataset subdirectory.'''

    fname = os.path.join(C.DATA_DIR, dataset)
    if not os.path.exists(fname):
        raise RuntimeError(f'Dataset does not exist: {fname}')
    
    fname = os.path.join(fname, subdir)
    if not os.path.exists(fname):
        raise RuntimeError(f'Subdirectory does not exist: {fname}')
        
    fname = os.path.join(fname, '*.png')
    return sorted(glob(fname))


class Pix2Patch(torch.nn.Module):
    def __init__(self, patch_size, input_dim=3) -> None:
        super().__init__()

        self.patch_size = patch_size

        if input_dim < 2:
            raise RuntimeError('Input dimension must be at least 2D.')

        # Adaptive kernel size
        # [1, ..., 1, patchsize, patchsize]
        self.kernel_size = [1] * input_dim
        self.kernel_size[-2:-1] = [patch_size, patch_size]

        # Averaging kernel with value 1/num_elements
        self.kernel = torch.ones(self.kernel_size)
        self.kernel = self.kernel / (patch_size ** 2)
        self.kernel.requires_grad = False

    def forward(self, pix_map):
        patch_map = F.conv2d(inputs=pix_map,
                            kernel=self.kernel,
                            stride=self.patch_size)
        return patch_map
