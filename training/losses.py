import torch.nn as nn

def _make_loss(loss_name: str) -> nn.Module:

    if loss_name == "ce":
        return nn.CrossEntropyLoss()

    else:
        raise ValueError(f"Loss function name must be one of the following options: ['ce'], got {loss_name}")