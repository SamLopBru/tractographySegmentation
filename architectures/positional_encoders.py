import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    pe: torch.Tensor # explicit typing for the buffer

    def __init__(self, model_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        if max_len > 10000.0:
            raise ValueError(f"max_len should be less than or equal to 10000. Got {max_len}.")

        pe = torch.zeros(max_len, model_dim)
        self.model_dim = model_dim
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() 
            * (-math.log(10000.0) / model_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe) # correctly moves pe with the module to the device
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)
