import os
import sys
import zipfile
from datetime import datetime
from glob import glob

import albumentations as A
import numpy as np
import torch
from torch.nn import functional as F

from configuration import CONSTANTS as C
from configuration import Configuration


def create_log_id(config:Configuration):
    '''Create a new logging id, containing timestamp and model name.'''
    timestamp = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    return f'{timestamp}--{config.model}'


def create_log_dir(config:Configuration):
    '''
    Create a new logging directory with an id containing timestamp and model name.
    
    Args:
        config: Configuration of current experiment.

    Returns:
        str: A directory where we can store logs. Raises an exception if the model directory already exists.    
    '''
    log_id = create_log_id(config)
    log_dir = os.path.join(C.RESULTS_DIR, log_id)

    # make experiments directory
    os.makedirs(C.RESULTS_DIR, exist_ok=True)

    if os.path.exists(log_dir):
        raise ValueError('Logging directory already exists {}'.format(log_dir))
    os.makedirs(log_dir)

    return log_dir, log_id


def export_cmd(fname):
    '''Stores command to a .sh file.'''
    if not fname.endswith('.sh'):
        fname += '.sh'

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

def to_str(sub:np.ndarray):
    '''Stream NumPy array to submission strings.'''
    for i_ind, p_ind_h, p_ind_w, pred  in sub:
        yield(f'{i_ind:03d}_{p_ind_h}_{p_ind_w},{pred}\n')

def to_csv(sub:np.ndarray, fname):
    '''Store NumPy array in submission file.'''
    with open(fname, 'w') as f:
        f.write('id,prediction\n')
        f.writelines(to_str(sub))


class ToFloatDual(A.DualTransform, A.ToFloat):
    '''Dual Transform of A.ToFloat.'''
    pass


class Pix2Patch(torch.nn.Module):
    def __init__(self, patch_size, input_dim=3) -> None:
        super().__init__()

        self.patch_size = patch_size

        if input_dim < 2:
            raise RuntimeError('Input dimension must be at least 2D.')

        # Adaptive kernel size
        # [1, ..., 1, patchsize, patchsize]
        self.kernel_size = [1] * input_dim
        self.kernel_size = [1, 1, patch_size, patch_size]

        # Averaging kernel with value 1/num_elements
        self.kernel = torch.ones(self.kernel_size, dtype=C.DTYPE)
        self.kernel = self.kernel / (patch_size ** 2)
        self.kernel.requires_grad = False

    def forward(self, pix_map:torch.Tensor):
        pix_map = pix_map.unsqueeze(1)
        patch_map = F.conv2d(input=pix_map,
                             weight=self.kernel,
                             stride=self.patch_size)
        return patch_map.squeeze(1)
