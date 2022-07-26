from copy import deepcopy
from typing import Any, Dict, Iterable, List, OrderedDict

import timm
import torch
from configuration import Configuration, create_optimizer
from timm.models.features import FeatureListNet
from torch import nn

from models.BaseModel import BaseModel


def invert_conv(conv: nn.Conv2d, **kwargs):
    # copy kwargs
    _kwargs = {k: conv.__dict__[k] for k in conv.__constants__}
    _kwargs['in_channels'] = conv.out_channels # invert channels
    _kwargs['out_channels'] = conv.in_channels
    _kwargs.update(kwargs)
    return nn.ConvTranspose2d(**_kwargs)

def adjust_conv(conv: nn.Conv2d, **kwargs):
    # copy kwargs
    _kwargs = {k: conv.__dict__[k] for k in conv.__constants__}
    _kwargs.pop('output_padding', None) # not available for Conv2d
    _kwargs.update(kwargs)
    return nn.Conv2d(**_kwargs)

def adjust_layer(layer: nn.Sequential, planes):
    new_layer = []
    for block in layer:
        t_block = type(block)
        new_block = t_block(inplanes=planes, planes=planes)
        new_layer.append(new_block)
    return nn.Sequential(*new_layer)

def get_first_conv(module:nn.Module):
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return child
    raise RuntimeError('First convolution not found')

def group_layer0(dec: FeatureListNet, flatten=False):
    layer0 = OrderedDict()
    for name in list(dec._modules.keys()):
        if name == 'layer1': break
        module = dec._modules.pop(name)
        if flatten and isinstance(module, nn.Sequential):
            for name, child in module.named_children():
                layer0[name] = child
            # layer0 |= module.named_children() # in python3.9
        else:
            layer0[name] = module
    dec._modules['layer0'] = nn.Sequential(*[nn.Sequential(layer0)])
    dec._modules.move_to_end('layer0', last=False)


class SkipConnection(nn.Module):
    def __init__(self, in_channels:int, skip_channels:int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = in_channels + skip_channels

    def forward(self, x:torch.Tensor, skip:torch.Tensor):
        return torch.cat([x, skip], dim=1)
        
    def extra_repr(self):
        return f'in={self.in_channels}, skip={self.skip_channels}, out={self.out_channels}'


class SkipLayer(nn.Module):
    def __init__(self, layer:nn.Module, index:int, channels:List[int]):
        super().__init__()
        self.index = index
        self.in_channels = channels[index]
        self.skip_channels = channels[index+1]
        self.out_channels = channels[index+1]

        # add deconvolution
        self.deconv = invert_conv(layer[0].conv1, 
                                    in_channels=self.in_channels, 
                                    out_channels=self.out_channels,
                                    stride=2, output_padding=1)
        # add skip connection
        self.skip = SkipConnection(in_channels=self.deconv.out_channels,
                                    skip_channels=self.skip_channels)
        self.conv1x1 = nn.Conv2d(in_channels=self.skip.out_channels,
                                    out_channels=self.out_channels,
                                    kernel_size=1)

        # add layer and update first convolution increase inpanes & not downsample
        self.layer = layer
        self.layer[0].conv1 = adjust_conv(layer[0].conv1, stride=1,
                                    in_channels=self.out_channels)
        if self.layer[0].downsample: # increase shortcut inpanes without downsampling 
            self.layer[0].downsample[1] = adjust_conv(self.layer[0].downsample[1], 
                                    in_channels=self.out_channels)
            self.layer[0].downsample._modules.pop('0') # remove avgpool downsampling

    def forward(self, x:torch.Tensor, skip:torch.Tensor):
        x = self.deconv(x)
        x = self.skip(x, skip)
        x = self.conv1x1(x)
        x = self.layer(x)
        return x


class FeatureListEncoder(nn.Module):
    def __init__(self, name:str, pretrained:bool=False, stem_reduce:bool=True) -> None:
        super().__init__()
        self.model = timm.create_model(name, features_only=True, pretrained=pretrained)
        self.channels = self.model.feature_info.channels()
        self.stem_reduce = stem_reduce

        if not self.stem_reduce: # change stride of stem from 2 to 1 
            get_first_conv(self.model.conv1).stride = (1,1)
        else:
            self.channels[0] = 3
        
    def forward(self, x:List[torch.Tensor]):
        features = self.model(x)
        if self.stem_reduce:
            features = [x,] + features[1:]
        return features

class FeatureListDecoder(nn.ModuleDict):
    def __init__(self, encoder:FeatureListEncoder) -> None:
        super().__init__(encoder.model._modules)
        self.channels = encoder.channels
        
        # group initial modules into layer 0, remove layer4
        #del self.bn1, self.act1, self.maxpool
        self._modules.popitem()
        group_layer0(self, flatten=True)
        
        # reorder modules to 
        self._modules = OrderedDict(reversed(self._modules.items()))
        self.channels = self.channels[::-1]

        # create outputlayer as copy of layer1
        self.layer0 = adjust_layer(self.layer1, planes=self.channels[-1])

        # make skip layers
        self.layer3 = SkipLayer(self.layer3, index=0, channels=self.channels)
        self.layer2 = SkipLayer(self.layer2, index=1, channels=self.channels)
        self.layer1 = SkipLayer(self.layer1, index=2, channels=self.channels)
        self.layer0 = SkipLayer(self.layer0, index=3, channels=self.channels)
        
    def forward(self, xs:List[torch.Tensor]):
        x = xs.pop() # last feature is new input 
        for module in self._modules.values():
            skip = xs.pop()
            x = module(x, skip)
        # TODO: is this memory optimal? are the unbound tensors deleted?
        return x


class Resnet(BaseModel):
    def __init__(self, config: Configuration):
        super().__init__(config)

        name = config.model.lower()
        if name[-1] != 'd':
            print('Warning: Currently only Resnets with deep stems are supported')

        ## create encoder
        #self.encoder = timm.create_model(name, features_only=True, pretrained=config.pretrained)
        self.encoder = FeatureListEncoder(name, config.pretrained, stem_reduce=False)
        self.out_channels = self.encoder.channels[-1]

        ## create decoder
        self.decoder = False
        if config.model_out == 'pixel':
            self.decoder = FeatureListDecoder(deepcopy(self.encoder))
            self.out_channels = self.decoder.channels[-1]
        
        self.head = nn.Conv2d(self.out_channels, 1, 1)

    def configure_optimizers(self):
        params = list(self.head.parameters())
        if self.decoder:
            params += list(self.decoder.parameters())
        t_opt, kwargs = create_optimizer(self.config)
        return t_opt(params, **kwargs)

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.config.freeze_epochs:
            param_group = dict(params=self.encoder.parameters())
            self.optimizers().add_param_group(param_group)
        return 

    def forward(self, batch: torch.Tensor):
        n_samples, n_channels, in_size, in_size = batch.shape

        batch = self.encoder(batch)
        if self.decoder: # decode if available
            batch = self.decoder(batch)
        else:
            batch = batch[-1] # last feature from pyramid

        batch = self.head(batch)
        return batch #batch.reshape(n_samples, 1, self.out_size, self.out_size)

    