import torch
from torch import nn
from torch.nn import functional as F


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


class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha = None, threshold: float = 0.5) -> None:
        super().__init__()
        self.gamma = gamma
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_class = (target > self.threshold)
        alpha = self.alpha or target_class.sum() / target_class.numel()

        # compute weights
        self.weight = torch.zeros_like(input)
        self.weight[target_class] = 1 - alpha   # invert weight of class to rebalance
        self.weight[~target_class] = alpha      # invert weight of class to rebalance
        
        ce_terms =( -(1-input).pow(self.gamma) * (  target) * torch.clamp(torch.log(  input), min=-100) 
                    -(  input).pow(self.gamma) * (1-target) * torch.clamp(torch.log(1-input), min=-100))
        return torch.sum(self.weight * ce_terms)