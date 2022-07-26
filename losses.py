import torch
from torch import nn
from torch.nn import functional as F


class BalancedBCELoss(nn.Module):
    def __init__(self, alpha: float = None, threshold: float = 0.5, reduction: str = 'meanw') -> None:
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha
        self.reduction = reduction


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_class = (target > self.threshold)
        N = target_class.numel()        # total number of targets
        Np = target_class.sum()         # number of targets in positive class
        Nn = N - Np                     # number of targets in negative class
        alpha = self.alpha or Np / N    # predefined or batchwise class frequency
        
        # compute the binary crossentropy
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

        # compute the balancing
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


class FocalLoss(BalancedBCELoss):
    def __init__(self, gamma: float, alpha: float = None, threshold: float = 0.5, reduction='meanw') -> None:
        super().__init__(alpha=alpha, threshold=threshold, reduction=reduction)
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_class = (target > self.threshold)
        N = target_class.numel()        # total number of targets
        Np = target_class.sum()         # number of targets in positive class
        Nn = N - Np                     # number of targets in negative class
        alpha = self.alpha or Np / N    # predefined or batchwise class frequency

        # compute logsigmoid inputs
        log_input = F.logsigmoid(input)
        log_input_inv = F.logsigmoid(-input)
        input = torch.exp(log_input)

        # compute the focal binary cross entropy
        loss = torch.zeros_like(input)
        loss -= (1-input).pow(self.gamma) *     target * log_input
        loss -=     input.pow(self.gamma) * (1-target) * log_input_inv

        # compute the balancing
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