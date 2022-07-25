import typing
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional


class BalancedBCELoss(nn.BCELoss):
    def __init__(self, alpha = None, threshold: float = 0.5, reduction: str = 'mean') -> None:
        super().__init__(reduction=reduction)
        self.threshold = threshold
        self.alpha = alpha


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_class = (target > self.threshold)
        alpha = self.alpha or target_class.sum() / target_class.numel()
        
        # compute weights
        self.weight = torch.zeros_like(input)
        self.weight[target_class] = 1 - alpha   # invert weight of class to rebalance
        self.weight[~target_class] = alpha      # invert weight of class to rebalance
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)

