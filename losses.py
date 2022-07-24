import typing
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional


class BalancedBCELoss(nn.BCELoss):
    def __init__(self, threshold: float = 0.5,
                    size_average=None, 
                    reduce=None, 
                    reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.threshold = threshold

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_class = (target > self.threshold)
        target_weight = target[target_class].numel() / target.numel()

        # compute weights
        self.weight = torch.zeros_like(input)
        self.weight[target_class] = target_weight
        self.weight[~target_class] = 1 - target_weight
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)

