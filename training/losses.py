import torch.nn as nn
import torch

def _make_loss(loss_name: str, device: torch.device) -> nn.Module:

    if loss_name == "ce":
        return nn.CrossEntropyLoss()

    else:
        raise ValueError(f"Loss function name must be one of the following options: ['ce'], got {loss_name}")