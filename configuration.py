'''
Define common constants that are used globally.

Also define a configuration object whose parameters can be set via the command line or loaded from an existing
json file. Here you can add more configuration parameters that should be exposed via the command line. In the code,
you can access them via `config.your_parameter`. All parameters are automatically saved to disk in JSON format.

Copyright ETH Zurich, Manuel Kaufmann & Felix Sarnthein

'''
import argparse
import json
import os
import pprint
from typing import Iterable, Union
from typing_extensions import Self

import torch

class Constants(object):
    '''
    This is a singleton.
    '''
    class __Constants:
        def __init__(self):
            # Environment setup.
            self.DTYPE = torch.float32
            self.IMG_SIZE = 400
            self.PATCH_SIZE = 16
            self.THRESHOLD = 0.25
            
            # Get directories from os.environ
            try: 
                self.DATA_DIR = os.environ['CIL_DATA']
                self.RESULTS_DIR = os.environ['CIL_RESULTS']
            except KeyError:
                raise RuntimeError(
                    '''Please configure the environment variables: 
                    - CIL_DATA: path to datasets
                    - CIL_RESULTS: path to store results''')

            # Print device
            if torch.cuda.is_available():
                GPU = torch.cuda.get_device_name(torch.cuda.current_device())
                print(f'Torch running on {GPU}...' , flush=True)
            else:
                print(f'Torch running on CPU...', flush=True)

    instance = None

    def __new__(cls, *args, **kwargs):
        if not Constants.instance:
            Constants.instance = Constants.__Constants()
        return Constants.instance

    def __getattr__(self, item):
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)

CONSTANTS = Constants()


class Configuration(object):
    '''Configuration parameters exposed via the commandline.'''

    def __init__(self, adict:dict):
        self.__dict__.update(adict)

    def __str__(self):
        return pprint.pformat(vars(self), indent=4, sort_dicts=False)

    @staticmethod
    def parser(pre_parser : argparse.ArgumentParser = None) -> argparse.ArgumentParser:

        # Argument parser.
        parents = [] if pre_parser is None else [pre_parser]
        parser = argparse.ArgumentParser(parents=parents,  
                    formatter_class=argparse.RawTextHelpFormatter)

        # General.
        general = parser.add_argument_group('General')
        general.add_argument('--name', type=str, default=None, 
                            help='Run name for Weights & Biases logger.')
        general.add_argument('--n_workers', type=int, default=4, 
                            help='Number of parallel threads for data loading.')
        general.add_argument('--seed', type=int, default=None,
                            help='Random number generator seed.')
        general.add_argument('--log_every', type=int, default=1,
                            help='Log every so many steps.')
        general.add_argument('--val_every', type=int, default=1,
                             help='Check val every n train epochs')
        general.add_argument('--force_cpu', action='store_true', default=False,
                            help='Force training on CPU instead of GPU.')

        # Data.
        data = parser.add_argument_group('Data')
        data.add_argument('--train_dir', type=str, default='gmaps',
                          help='Training dataset directory name under $CIL_DATA')
        data.add_argument('--valid_dir', type=str, default='training',
                          help='Validation dataset directory name under $CIL_DATA')
        data.add_argument('--test_dir', type=str, default='test',
                          help='Test dataset directory name under $CIL_DATA')
        data.add_argument('--bs_train', type=int, default=16, 
                            help='Batch size for the training set.')
        data.add_argument('--bs_eval', type=int, default=16, 
                            help='Batch size for valid/test set.')
        data.add_argument('--normalize', action='store_true', default=False)
        data.add_argument('--aug', type=str, nargs='*', default=[],
                            help='List of named augmentations to apply to training data.')

        # Model.
        model = parser.add_argument_group('Model')
        model.add_argument('--model', type=str, default='LinearConv', 
                            help='Defines the model to train on.')
        model.add_argument('--load_model', type=str, default=None,
                            help='Load model from given checkpoint.')
        model.add_argument('--pretrained', action='store_true', default=False, 
                            help='Use a pretrained model from an external library.')
        model.add_argument('--freeze_epochs', type=int, default=0,
                            help='Number of epochs to freeze if supported by model.')
        model.add_argument('--model_out', type=str, choices={'pixel', 'patch'}, default='pixel',
                            help='Output features of model. Some models might not support pixelwise.')
        model.add_argument('--loss_in', type=str, choices={'pixel', 'patch'}, default='patch',
                            help='Input features of loss. Pixel activations are automatically averaged to patches.')
        model.add_argument('--loss', type=str, choices={'bce', 'bbce', 'focal'}, default='bce', 
                            help='Type of loss for training.')
        model.add_argument('--bbce_alpha', type=float, default=None,
                            help='Dataset imbalance ratio, estimated batchwise by default.')
        model.add_argument('--focal_gamma', type=float, default=2,
                            help='Focus factor of focal loss, the paper proposes gamma=2.')

        
        # Training configurations.        
        training = parser.add_argument_group('Training')
        training.add_argument('--n_epochs', type=int, default=50, 
                            help='Number of epochs to train for.')
        training.add_argument('--opt', type=str, choices={'adamw', 'adam', 'sgd'}, default='adamw', 
                            help='Optimizer to use for training.')
        training.add_argument('--lr', type=float, default=None, 
                            help='Learning rate for optimizer.')
        training.add_argument('--wd', type=float, default=None, 
                            help='Weight decay for optimizer.')
        training.add_argument('--precision', type=int, default=32,
                              help='Precision bits for floating points during training')

        return parser
    
    @staticmethod
    def get_default():
        parser = Configuration.parser()
        defaults = parser.parse_args([])
        return Configuration(vars(defaults))

    @staticmethod
    def from_json(json_path:str, default_config=None):
        '''Load configurations from a JSON file.'''

        # Get default configuration
        if default_config is None:
            default_config = Configuration.get_default()

        # Load configuration from json file
        with open(json_path, 'r') as f:
            json_config = json.load(f) 

        # Overwrite defaults
        default_config.update(json_config)

        return default_config

    @staticmethod
    def parse_cmd():
        '''Loading configuration according to priority:
        1. from commandline arguments
        2. from JSON configuration file
        3. from parser default values.'''
        
        # Initial parser.
        pre_parser = argparse.ArgumentParser(add_help=False)

        pre_parser.add_argument('--from_json', type=str,
                            help=Configuration.parse_cmd.__doc__)

        # Argument parser.
        parser = Configuration.parser(pre_parser)

        # 1. Get default configurations
        config = Configuration.get_default()

        # 2. Overwrite with defaults with JSON config
        args = parser.parse_args() # irgnore args except --from_json
        if args.from_json is not None:
            config = Configuration.from_json(args.from_json, config)

        # 3. Overwrite JSON config with remaining cmd args
        parser.parse_args(namespace=config)

        return config


    def to_json(self, json_path:str):
        '''Dump configurations to a JSON file.'''
        with open(json_path, 'w') as f:
            s = json.dumps(vars(self), indent=2)
            f.write(s)

    def update(self, adict:Union[dict, Self]):
        if isinstance(adict, Iterable):
            self.__dict__.update(adict)
        else:
            self.__dict__.update(adict.__dict__)


def create_model(config:Configuration):
    '''
    This is a helper function that can be useful if you have several model definitions that you want to
    choose from via the command line.
    '''
    if config.model == 'LinearFC':
        from models.LinearFC import LinearFC
        return LinearFC(config)

    if config.model == 'LinearConv':
        from models.LinearConv import LinearConv
        return LinearConv(config)

    if config.model == 'BaselineUNet':
        from models.BaselineUNet import BaselineUNet
        return BaselineUNet(config)

    if config.model[:6] == 'Resnet':
        from models.ResNet import Resnet
        return Resnet(config)


    raise RuntimeError(f'Unkown model name: {config.model}')


def create_loss(config:Configuration):
    '''
    This is a helper function that can be useful if you have losses that you want to
    choose from via the command line.
    '''
    if config.loss == 'bce':
        return torch.nn.BCEWithLogitsLoss()
    
    if config.loss == 'bbce':
        from losses import BalancedBCELoss
        thresh = CONSTANTS.THRESHOLD if (config.loss_in == 'patch') else 0.5
        alpha = config.bbce_alpha if 'bbce_alpha' in config.__dict__ else None # backwards compatibility
        return BalancedBCELoss(alpha=alpha, threshold=thresh)

    if config.loss == 'focal':
        from losses import FocalLoss
        thresh = CONSTANTS.THRESHOLD if (config.loss_in == 'patch') else 0.5
        return FocalLoss(gamma=config.focal_gamma, alpha=config.bbce_alpha, threshold=thresh)

    raise RuntimeError(f'Unkown loss name: {config.loss}')


def create_optimizer(config:Configuration):
    '''
    This is a helper function that can be useful if you have optimizers that you want to
    choose from via the command line.
    '''
    kwargs = dict()
    if config.lr:
        kwargs['lr'] = config.lr
    if config.wd:
        kwargs['weight_decay'] = config.wd

    if config.opt == 'adamw':
        return torch.optim.AdamW, kwargs

    if config.opt == 'adam':
        return torch.optim.Adam, kwargs

    if config.opt == 'sgd':
        return torch.optim.SGD, kwargs

    raise RuntimeError(f'Unkown optimizer name: {config.opt}')


def create_augmentation(config:Configuration):
    '''
    This is a helper function that can be useful if you have augmentations that you want to
    choose from via the command line.
    '''

    from albumentations import NoOp
    transforms = [NoOp()]

    # iterate through list and add totransforms
    for aug_spec in config.aug:

        if aug_spec == 'aug_with_crop':
            from augmentations import aug_with_crop
            transforms.append(aug_with_crop())
            continue

        if aug_spec == 'aug_without_crop':
            from augmentations import aug_without_crop
            transforms.append(aug_without_crop())
            continue

        if aug_spec == 'resize384':
            from augmentations import resize
            transforms.append(resize(384))
            continue

        if aug_spec == 'crop128':
            from augmentations import crop
            transforms.append(crop(128))
            continue

        if aug_spec == 'crop256':
            from augmentations import crop
            transforms.append(crop(256))
            continue
        
        if aug_spec != '':  # fall through case
            raise RuntimeError(f'Unknown augmentations {aug_spec}')

    from albumentations import Compose
    return Compose(transforms)