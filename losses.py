import torch
from torch import nn
from torch.nn import functional as F


class BalancedBCELoss(nn.BCEWithLogitsLoss):
    def __init__(self, alpha = None, threshold: float = 0.5, reduction: str = 'mean') -> None:
        super().__init__(reduction=reduction)
        self.threshold = threshold
        self.alpha = alpha


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_class = (target > self.threshold)
        alpha = self.alpha or target_class.sum() / target_class.numel()
        
        # compute weights
        weight = torch.zeros_like(input)
        weight[target_class] = 1 - alpha   # invert weight of class to rebalance
        weight[~target_class] = alpha      # invert weight of class to rebalance
        return F.binary_cross_entropy_with_logits(input, target, weight=weight, reduction=self.reduction)


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
        weight = torch.zeros_like(input)
        weight[target_class] = 1 - alpha   # invert weight of class to rebalance
        weight[~target_class] = alpha      # invert weight of class to rebalance

        log_preds = F.log_softmax(input)
        preds = torch.exp(log_preds)

        ce_terms = -((1-preds).pow(self.gamma) * (  target) * log_preds 
                     + (preds).pow(self.gamma) * (1-target) * log_preds)
        return torch.sum(weight * ce_terms)