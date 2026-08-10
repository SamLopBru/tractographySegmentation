import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, LSTM
from torch.nn.utils.rnn import pack_padded_sequence
from positional_encoders import SinusoidalPositionalEncoding


class ClassifierHead(nn.Module):

    def __init__(self,
                model_dim: int,
                num_classes: int,
                dropout_p: float):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(model_dim, num_classes)
        )
    def forward(self, x):
        return self.classifier(x)

class TransformerEncoder(nn.Module):
    """
    Transformer based encoder for classifying streamlines into bundles

    Input: Streamlines of shape (B, max_seq_len, 5)
    Output: Class logits of shape (B, num_classes)
    """
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
            positional_encoding: str = "sinusoidal",
            normalization: str = "layernorm" # Normalization layer for the transformer encoder
    ):
        super().__init__()
        
        # Validate the pooling strategy and the positional encoding
        if pooling_strategy not in ["mean", "max", "cls"]:
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}. Must be one of ['mean', 'max', 'cls'].")
        
        if positional_encoding not in ["sinusoidal", "rope"]:
            raise ValueError(f"Invalid positional encoding: {positional_encoding}. Must be one of ['sinusoidal', 'rope'].")

        if normalization not in ["layernorm", "rmsnorm"]:
            raise ValueError(f"Invalid normalization: {normalization}. Must be one of ['layernorm', 'rmsnorm'].")

        self.model_dim = model_dim
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout
        self.pooling_strategy = pooling_strategy
        self.input_projection = nn.Linear(input_dim, model_dim)

        # Initialize cls token if pooling strategy is 'cls'
        if self.pooling_strategy == "cls":
            self.cls_token = nn.Parameter(torch.empty(1, 1, model_dim)) # does not mind if it's empty because it's going to be reinitialized
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        if positional_encoding == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(self.model_dim, dropout=self.dropout_p)
        
        # Create the transformer encoder layers
        self.encoder_layers = TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout_p,
                activation='gelu',
                norm_first=True,
                batch_first=True
            )
        # Replace LayerNorm with RMSNorm 
        if normalization == "rmsnorm":
            self.encoder_layers.norm1 = nn.RMSNorm(model_dim, eps=1e-6)  # pyright: ignore[reportAttributeAccessIssue]
            self.encoder_layers.norm2 = nn.RMSNorm(model_dim, eps=1e-6)  # pyright: ignore[reportAttributeAccessIssue]

        # Create the transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layers,
            num_layers=num_layers,
            enable_nested_tensor=False # maybe this is silently ignored because norm_first=True
        )

        # Create the classifier head
        self.classifier = ClassifierHead(model_dim=model_dim,
                                        num_classes=num_classes,
                                        dropout_p=dropout)

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

        batch_size, max_seq_len, _ = x.shape  # (B, max_seq_len, 5)

        x = self.input_projection(x) # (B, max_seq_len, d_model)

        # Create padding mask from lengths
        indices = torch.arange(max_seq_len, device=x.device).unsqueeze(0) # (1, max_seq_len)
        padding_mask = indices >= lengths.unsqueeze(1) # (B, max_seq_len) when greater than the real length, the value is True

        if self.pooling_strategy == "cls":
            # Expand the cls token to match the batch size
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1) # (B, max_seq_len +1, d_model)
            
            # Append a False for the padding mask 
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)  # (B, 1)
            padding_mask = torch.cat((cls_mask, padding_mask), dim=1)  # (B, max_seq_len + 1)
        
        
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask) # (B, max_seq_len [+1 if cls], d_model)
        
        if self.pooling_strategy == "cls":
            x_pooled = x[:, 0]
        elif self.pooling_strategy == "mean":
            mask = ~padding_mask.unsqueeze(-1) # (B, max_seq_len, 1)
            x_pooled = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1) # clamp to avoid dividing by 0
        elif self.pooling_strategy == "max":
            x = x.masked_fill(padding_mask.unsqueeze(-1), float('-inf')) # (B, max_seq_len, 1)
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
    
class LSTMEncoder(nn.Module):

    def __init__(self,
                input_dim: int = 5,
                hidden_dim: int = 128,
                num_layers: int = 2,
                bias: bool = True, # need to be True for _init_weights to work
                batch_first: bool = True,
                dropout: float = 0.1,
                bidirectional: bool = True,
                num_classes: int = 32,
                pooling_strategy: str = "mean"):
        
        super().__init__()

        if pooling_strategy not in ["mean", "max", "last"]:
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}. Must be one of ['mean', 'max', 'last'].")

        self.pooling_strategy = pooling_strategy

        self.lstm = LSTM(input_dim,
                        hidden_dim,
                        num_layers,
                        bias=bias,
                        batch_first=batch_first,
                        dropout=dropout,
                        bidirectional=bidirectional
        )

        self.classifier = ClassifierHead(model_dim=hidden_dim * 2 if bidirectional else hidden_dim,
                                        num_classes=num_classes,
                                        dropout_p=dropout)

        self._init_weights()

    def _init_weights(self):
        # all the "param" variables are the weights and biases of the LSTM layers in one sigle 1-D tensor
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

            elif "bias" in name:
                nn.init.zeros_(param)

                hidden_size = param.shape[0] // 4

                # forget gate initialize to 1 to retain information at the beginning of training: sigmoid(1.0) = 0.731
                # if the forget gate is initialized to 0, the LSTM will forget everything at the beginning of training: sigmoid(0.0) = 0.5
                # (https://arxiv.org/abs/1909.09502)
                with torch.no_grad():
                    param[hidden_size:2 * hidden_size].fill_(1.0)

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_embeddings(self, x, lengths) -> torch.Tensor:
        
        packed_input = nn.utils.rnn.pack_padded_sequence(input=x, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, (h_n, c_n) = self.lstm(packed_input)

        if self.pooling_strategy == "last":
            if self.lstm.bidirectional:
                embeddings = torch.cat((h_n[-2], h_n[-1]), dim=1)  # last hidden states from both directions
            else:
                embeddings = h_n[-1]  # last hidden state from the last layer

        else:
            unpacked_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            mask = torch.arange(unpacked_output.size(1), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

            if self.pooling_strategy == "mean":
                embeddings = (unpacked_output * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)

            else: # pooling_strategy == "max":
                embeddings = unpacked_output.masked_fill(~mask.unsqueeze(-1), float('-inf')).max(dim=1)[0]
            
        return embeddings

    def forward(self, x, lengths) -> torch.Tensor:
        embeddings = self.get_embeddings(x, lengths)
        logits = self.classifier(embeddings)
        return logits


# model = LSTMEncoder()

# print(model)