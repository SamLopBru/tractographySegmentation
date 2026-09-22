import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    """"
    Focal loss for multi-class classification. Following: https://arxiv.org/abs/1708.02002
    Args:
        alpha (float): Weighting factor for the class balance. Default is 1.0
        gamma (float): Focusing parameter to reduce the relative loss for well-classified examples.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none') # With none reduction it returns the loss for each sample in the batch

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def _make_loss(loss_name: str, device: torch.device, **loss_params) -> nn.Module:

    if loss_name == "ce":
        return nn.CrossEntropyLoss(**loss_params).to(device)

    elif loss_name == "focal":
        return FocalLoss(**loss_params).to(device)

    else:
        raise ValueError(f"Loss function name must be one of the following options: ['ce', 'focal'], got {loss_name}")