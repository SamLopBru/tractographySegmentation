from dataclasses import dataclass


@dataclass
class GlobalConfiguration:

    # Paths
    trainLoader_path: str = ""
    valLoader_path: str = ""

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

    # Training hyperparameters
    # use_amp: bool = True
    learning_rate: float = 0.1
    weight_decay: float = 0.05
    num_epochs: int = 40
    patience: int = 10
    accumulation_steps: int = 2
    validate_every = 2
    warmup_steps = 512

    def __post_init__(self):
        assert self.encoder_type in ['transformer', 'lstm'], f"Encoder type must be one of the following options: ['transformer', 'lstm'], got {self.encoder_type}"

        assert self.pooling_strategy in ['cls', 'max', 'mean'], f"Pooling strategy must be one of the following options: ['max', 'mean', 'cls'], got {self.pooling_strategy}"

        assert self.loss_fn_type in ['ce', 'focal'], f"Loss function type must be one of the following options: ['ce', 'focal'], got {self.loss_fn_type}"
