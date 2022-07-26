import torch
from torch import nn

from configuration import CONSTANTS as C
from models.BaseModel import BaseModel

class Block(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(out_ch),
                                   nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                   nn.ReLU())

    def forward(self, x):
        return self.block(x)

        
class BaselineUNet(BaseModel):
    '''Copy of the UNet which is proposed in the tutorial.'''
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, config, chs=(3,64,128,256,512,1024)):
        super().__init__(config)

        enc_chs = chs  # number of channels in the encoder
        self.enc_blocks = nn.ModuleList([Block(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])])  # encoder blocks
        self.pool = nn.MaxPool2d(2)  # pooling layer (can be reused as it will not be trained)
        if config.model_out == 'pixel':
            dec_chs = chs[::-1][:-1]  # number of channels in the decoder
            self.upconvs = nn.ModuleList([nn.ConvTranspose2d(in_ch, out_ch, 2, 2) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # deconvolution
            self.dec_blocks = nn.ModuleList([Block(in_ch, out_ch) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # decoder blocks
            self.head = nn.Sequential(nn.Conv2d(dec_chs[-1], 1, 1)) # 1x1 convolution for producing the output
            # => Sequential for backward compatibility, when loading checkpoints
        else:
            self.head = nn.Sequential(nn.Conv2d(enc_chs[-1], 1, 1)) # 1x1 convolution for producing the output
            # => Sequential for backward compatibility, when loading checkpoints

        

    def forward(self, x):
        n_samples, n_channels, in_width, in_height = x.shape

        # encode
        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution
        x = self.enc_blocks[-1](x)
        # decode
        if self.config.model_out == 'pixel':
            for block, upconv, feature in zip(self.dec_blocks, self.upconvs, enc_features[::-1]):
                x = upconv(x)  # increase resolution
                x = torch.cat([x, feature], dim=1)  # concatenate skip features
                x = block(x)  # pass through the block
        x = self.head(x)  # reduce to 1 channel

        return x #x.reshape(n_samples, 1, self.out_size, self.out_size)
