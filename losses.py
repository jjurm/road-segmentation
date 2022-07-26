import torch
from torch import nn
from torch.nn import functional as F


class BalancedBCELoss(nn.BCEWithLogitsLoss):
    def __init__(self, alpha = None, threshold: float = 0.5, reduction: str = 'meanw') -> None:
        super().__init__(reduction=reduction)
        self.threshold = threshold
        self.alpha = alpha


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_class = (target > self.threshold)
        N = target_class.numel()        # total number of targets
        Np = target_class.sum()         # number of targets in positive class
        Nn = N - Np                     # number of targets in negative class
        alpha = self.alpha or Np / N    # predefined or batchwise class frequency
        
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        loss[target_class] *= 1 - alpha # weigh positive targets with negative class frequency
        loss[~target_class] *= alpha    # weigh negative targets with positive class frequency

        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.sum() / N
        if self.reduction == 'meanw':
            # weigh the mean as if there were alpha samples in the positive class
            N_weighted = Np * (1-alpha) + Nn * alpha
            loss = loss.sum() / N_weighted
        return loss


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

        log_preds = F.logsigmoid(input)
        preds = torch.exp(log_preds)

        ce_terms = -((1-preds).pow(self.gamma) * (  target) * log_preds 
                     + (preds).pow(self.gamma) * (1-target) * log_preds)
        return torch.sum(weight * ce_terms)