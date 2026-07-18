import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .positional_encoders import SinusoidalPositionalEncoding

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int = 5,
            model_dim: int = 128,
            dim_feedforward: int = 512,
            num_heads: int = 8,
            num_layers: int = 4,
            dropout: float = 0.1,
            num_classes: int = 32,
            pooling_strategy: str = "mean",
            positional_encoding: str = "sinusoidal"
    ):
        super().__init__()
        
        # Validate the pooling strategy and the positional encoding
        if pooling_strategy not in ["mean", "max", "cls"]:
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}. Must be one of ['mean', 'max', 'cls'].")
        
        if positional_encoding not in ["sinusoidal", "rope"]:
            raise ValueError(f"Invalid positional encoding: {positional_encoding}. Must be one of ['sinusoidal', 'rope'].")

        self.model_dim = model_dim
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.pooling_strategy = pooling_strategy
        self.input_projection = nn.Linear(input_dim, model_dim)

        # Initialize cls token if pooling strategy is 'cls'
        if self.pooling_strategy == "cls":
            self.cls_token = nn.Parameter(torch.rand(1, 1, model_dim))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        if positional_encoding == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(self.model_dim, dropout=self.dropout)
        
        # Create the transformer encoder layers
        self.encoder_layers = TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation='gelu',
                norm_first=True,
                batch_first=True
            )

        # Create the transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layers,
            num_layers=num_layers,
            enable_nested_tensor=True
        )

        # Create the classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(model_dim, num_classes)
        )

        self._init_weights()
        

    def _init_weights(self):
        """
        Initialize weights for the model using Xavier uniform initialization for 
        Linear layers and zeros for biases.
        """
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def get_embeddings(self,
                    x: torch.Tensor,
                    lengths: torch.Tensor # (B,)
                    ) -> torch.Tensor:
        """
        Get the embeddings from the transformer encoder. This function is separted from
        the forward pass to allow for flexibility in using the embeddings for different tasks (such
        as contrastive learning or t-SNE visualization).
        """

        batch_size, seq_len, _ = x.shape  # (B, seq_len, 5)

        x = self.input_projection(x) # (B, seq_len, d_model)

        # Create padding mask from lengths
        indices = torch.arange(seq_len, device=x.device).unsqueeze(0) # (1, seq_len)
        padding_mask = indices >= lengths.unsqueeze(1) # (B, seq_len) when greater than the real length, the value is True

        if self.pooling_strategy == "cls":
            # Expand the cls token to match the batch size
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1) # (B, seq_len +1, d_model)
            
            # Append a False for the padding mask 
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)  # (B, 1)
            padding_mask = torch.cat((cls_mask, padding_mask), dim=1)  # (B, seq_len + 1)
        
        
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask) # (B, seq_len [+1 if cls], d_model)
        
        if self.pooling_strategy == "cls":
            x_pooled = x[:, 0]
        elif self.pooling_strategy == "mean":
            mask = ~padding_mask.unsqueeze(-1) # (B, seq_len, 1)
            x_pooled = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1) # clamp to avoid dividing by 0
        elif self.pooling_strategy == "max":
            x = x.masked_fill(padding_mask.unsqueeze(-1), float('-inf')) # (B, seq_len, 1)
            x_pooled = x.max(dim=1)[0].clamp(min=-1e9)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return x_pooled # (B, d_model)
    
    def forward(self,
                x: torch.Tensor,
                lengths: torch.Tensor
                ) -> torch.Tensor:
        
        x_pooled = self.get_embeddings(x, lengths)
        logits = self.classifier(x_pooled)

        return logits
    
