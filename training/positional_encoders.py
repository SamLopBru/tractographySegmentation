import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        if max_len > 10000.0:
            raise ValueError(f"max_len should be less than or equal to 10000. Got {max_len}.")

        self.pe = torch.zeros(max_len, model_dim)
        self.model_dim = model_dim
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() 
            * (-math.log(10000.0) / model_dim)
        )

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', self.pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)
