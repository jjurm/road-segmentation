from torchvision.models import resnet18, resnet34, resnet50
from configuration import Configuration
from models.BaseModel import BaseModel
import torch
from torch import nn


class ResNetEncoder(BaseModel):
    def __init__(self, config: Configuration):
        super().__init__(config)

        if config.model_out == 'pixels':
            raise RuntimeError('ResNetEncoder only supports patchwise predictions.')

        if config.model == 'ResNet18':
            self.resnet = resnet18(pretrained=True)

        if config.model == 'ResNet34':
            self.resnet = resnet34(pretrained=True)

        if config.model == 'ResNet50':
            self.resnet = resnet50(pretrained=True)

        self.head = nn.Sequential(nn.Conv2d(256, 1, 1), nn.Sigmoid())


    def forward(self, x):
        n_samples, n_channels, in_width, in_height = x.shape
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)

        if self.trainer.current_epoch < 1:
            x = x.detach() # freeze backbone during first epoch

        x = self.head(x)
        return x.reshape(n_samples, self.out_size, self.out_size)






    