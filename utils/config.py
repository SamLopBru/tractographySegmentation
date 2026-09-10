from dataclasses import dataclass
from typing import Optional


@dataclass
class GlobalConfiguration:

    # Paths
    trainLoader_path: str = "sequences/testset"
    valLoader_path: str = "sequences/testset"

    # Encoder parameters
    encoder_type: str = 'transformer'
    input_dim: int = 5
    num_classes: int = 32
    model_dim: int = 128
    feedforward_dim : int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.2
    pooling_strategy: str = 'cls'
    positional_encoding: str = 'sinusoidal'
    loss_fn_type: str = "ce"
    layer_normalization: str = "layernorm"

    # Training hyperparameters
    # use_amp: bool = True
    batch_size: int = 1024
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    num_epochs: int = 40
    patience: int = 10
    accumulation_steps: int = 2
    validate_every: int = 2
    warmup_steps: int = 512

    num_workers: int = 4
    sampling_percentage_train: float = 0.1
    sampling_percentage_val: float = 0.1
    min_streamlines: int = 100
    max_streamlines: Optional[int] = None
    seed: int = 42

    # LSTM hyperaparameters
    bidirectional: bool = True

    def __post_init__(self):
        assert self.encoder_type in ['transformer', 'lstm', 'gru'], f"Encoder type must be one of the following options: ['transformer', 'lstm'], got {self.encoder_type}"

        assert self.pooling_strategy in ['cls', 'max', 'mean', 'last'], f"Pooling strategy must be one of the following options: ['max', 'mean', 'cls'], got {self.pooling_strategy}"

        assert self.loss_fn_type in ['ce', 'focal'], f"Loss function type must be one of the following options: ['ce', 'focal'], got {self.loss_fn_type}"

        assert self.layer_normalization in ['layernorm', 'rmsnorm'], f"Layer normalization must be one of the following options: ['layernorm', 'rmsnorm'], got {self.layer_normalization}"
