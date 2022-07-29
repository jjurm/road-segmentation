import os
import sys
import zipfile
from datetime import datetime
from glob import glob

import albumentations as A
import numpy as np
import torch
from torch.nn import functional as F

from PIL import Image, ImageFont, ImageDraw
from matplotlib.font_manager import findfont, FontProperties

from configuration import CONSTANTS as C
from configuration import Configuration

from torchvision.utils import make_grid

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

def to_str(sub:torch.Tensor):
    '''Stream NumPy array to submission strings.'''
    for i_ind, p_ind_h, p_ind_w, pred  in sub:
        yield(f'{int(i_ind):03d}_{int(p_ind_h)}_{int(p_ind_w)},{int(pred)}\n')

def to_csv(sub:np.ndarray, fname):
    '''Store NumPy array in submission file.'''
    with open(fname, 'w') as f:
        f.write('id,prediction\n')
        f.writelines(to_str(sub))


class ToFloatDual(A.DualTransform, A.ToFloat):
    '''Dual Transform of A.ToFloat: 
    While `A.ToFloat` only casts the image to float, 
    `ToFloatDual` casts both image and mask to float.'''
    pass


class Pix2Patch(torch.nn.Module):
    '''Transform Pixel to Patch Maps:
    This nn.Module takes a pixel map, computes the patch-wise averages
    and stores them in a patch map. This can be used to compute the ratio of active pixels
    in a segmentation map.'''
    def __init__(self, patch_size) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.kernel_size = [1, 1, patch_size, patch_size]

        # Summing kernel with value 1
        self.register_buffer('kernel', torch.ones(self.kernel_size, dtype=C.DTYPE))
        self.kernel.requires_grad = False
        self.num_elements = self.kernel.sum()

    def forward(self, pix_map:torch.Tensor) -> torch.Tensor:
        patch_map = F.conv2d(input=pix_map.reshape(-1, 1, *pix_map.shape[-2:]),
                             weight=self.kernel,
                             stride=self.patch_size) / self.num_elements
        return patch_map.reshape(*pix_map.shape[:-2], *patch_map.shape[-2:])

class ImageTextRenderer:
    def __init__(self, size=60):
        font_path = findfont(FontProperties(family='monospace'))
        self.font = ImageFont.truetype(font_path, size=size, index=0)
        self.size = size

    def print_gray(self, img_np_f, text, offs_xy, white=1.0):
        assert len(img_np_f.shape) == 2, "Image must be single channel"
        # print("shapee:", img_np_f.shape)
        # exit()
        img_pil = Image.fromarray(img_np_f, mode='F')
        ctx = ImageDraw.Draw(img_pil)
        step = self.size // 15
        for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            ctx.text((offs_xy[0] + step * dx, offs_xy[1] + step * dy), text, font=self.font, fill=0.0)
        ctx.text(offs_xy, text, font=self.font, fill=white)
        return np.array(img_pil)

    def print(self, img_np_f, text, offs_xy, **kwargs):
        if len(img_np_f.shape) == 3:
            for ch in range(img_np_f.shape[0]):
                img_np_f[ch] = self.print_gray(img_np_f[ch], text, offs_xy, **kwargs)
        else:
            img_np_f = self.print_gray(img_np_f, text, offs_xy, **kwargs)
        return img_np_f

_text_renderers = dict()

def get_text_renderer(size):
    if size not in _text_renderers:
        _text_renderers[size] = ImageTextRenderer(size)
    return _text_renderers[size]


def img_print(*args, **kwargs):
    size = kwargs['size']
    del kwargs['size']
    renderer = get_text_renderer(size)
    return renderer.print(*args, **kwargs)

def tensor_print(img, caption, **kwargs):
    if isinstance(caption, str) and len(caption.strip()) == 0:
        return img
    assert img.dim() == 4 and img.shape[1] in (1, 3), 'Expecting 4D tensor with RGB or grayscale'
    offset = min(img.shape[2], img.shape[3]) // 100
    img = img.cpu()
    offset = (offset, offset)
    size = min(img.shape[2], img.shape[3]) // 15
    for i in range(img.shape[0]):
        tag = (caption if isinstance(caption, str) else caption[i]).strip()
        if len(tag) == 0:
            continue
        img_np = img_print(img[i].numpy(), tag, offset, size=size, **kwargs)
        img[i] = torch.from_numpy(img_np)
    return img

def compose(list_triples):
    # list_triples = [(img[:1], c) for (img, c) in list_triples]
    vis = []
    for i, (img, caption) in enumerate(list_triples):
        vis.append(tensor_print(
            img.clone(), caption if type(caption) is list else [str(caption)] * img.shape[0])
        )

    vis = torch.cat(vis, dim=2)

    # N x 3 x H * N_MODS x W
    vis = make_grid(vis, nrow=max(len(list_triples), 1), padding=2, pad_value=0.5)
    return vis




    
###############################################################################
########################## PyTorch Pre-Release Code ###########################
###############################################################################
# Source: https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
import math
import warnings

import torch
from torch import Tensor


# These no_grad_* functions are necessary as wrappers around the parts of these
# functions that use `with torch.no_grad()`. The JIT doesn't support context
# managers, so these need to be implemented as builtins. Using these wrappers
# lets us keep those builtins small and re-usable.
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor: Tensor, mean: float = 0., std: float = 1., a: float = -2., b: float = 2.) -> Tensor:
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)