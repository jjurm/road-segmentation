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
        general.add_argument('--data_workers', type=int, default=4, 
                            help='Number of parallel threads for data loading.')
        general.add_argument('--print_every', type=int, default=200, 
                            help='Print stats to console every so many iters.')
        general.add_argument('--eval_every', type=int, default=400, 
                            help='Evaluate validation set every so many iters.')
        general.add_argument('--seed', type=int, default=None,
                            help='Random number generator seed.')

        # Data.
        data = parser.add_argument_group('Data')
        data.add_argument('--bs_train', type=int, default=16, 
                            help='Batch size for the training set.')
        data.add_argument('--bs_eval', type=int, default=16, 
                            help='Batch size for valid/test set.')

        # Model.
        model = parser.add_argument_group('Model')
        model.add_argument('--model', type=str, default='LinearConv', 
                            help='Defines the model to train on.')
        model.add_argument('--model_out', type=str, choices={'pixels', 'patches'}, default='pixels',
                            help='Output features of model. Some models might not support pixels.')
        model.add_argument('--loss_in', type=str, choices={'pixels', 'patches'}, default='patches',
                            help='Input features of loss. Pixel activations are automatically transformed to patches.')
        model.add_argument('--loss', type=str, choices={'bce'}, default='bce', 
                            help='Type of loss for training.')

        
        # Training configurations.        
        training = parser.add_argument_group('Training')
        training.add_argument('--n_epochs', type=int, default=50, 
                            help='Number of epochs to train for.')
        training.add_argument('--opt', type=str, choices={'adam', 'sgd'}, default='adam', 
                            help='Optimizer to use for training.')
        training.add_argument('--lr', type=float, default=0.001, 
                            help='Learning rate for optimizer.')

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

        # 1. Get defaults from parser
        config = Configuration.get_default()

        # 2. Overwrite with defaults with JSON config
        pre_args, remaining_argv = pre_parser.parse_known_args()
        if pre_args.from_json is not None:
            json_path = pre_args.from_json 
            config = Configuration.from_json(json_path, config)
            config.from_json = pre_args.from_json

        # 3. Overwrite JSON config with remaining cmd args
        parser.parse_args(remaining_argv, config)

        return config


    def to_json(self, json_path:str):
        '''Dump configurations to a JSON file.'''
        with open(json_path, 'w') as f:
            s = json.dumps(vars(self), indent=2)
            f.write(s)

    def update(self, adict:dict):
        self.__dict__.update(adict)


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
    
    if config.model == 'UNet':
        from models.UNet import UNet
        return UNet(config)

    raise RuntimeError('Unkown model name.')


def create_loss(config:Configuration):
    '''
    This is a helper function that can be useful if you have losses that you want to
    choose from via the command line.
    '''
    if config.loss == 'bce':
        return torch.nn.BCELoss()

    raise RuntimeError('Unkown loss name.')


def create_optimizer(model:torch.nn.Module, config:Configuration):
    '''
    This is a helper function that can be useful if you have optimizers that you want to
    choose from via the command line.
    '''
    if config.opt == 'adam':
        return torch.optim.Adam(model.parameters(), lr=config.lr)

    if config.opt == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=config.lr)

    raise RuntimeError('Unkown optimizer name.')
