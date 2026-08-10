import argparse
import os
import sys
import csv
from datetime import datetime

from typing import Optional, Protocol

from torch import nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from training.encoder import TransformerEncoder, LSTMEncoder
from utils.config import GlobalConfiguration
from utils.dataloader import StratifiedEpochSampler, StreamlineDataset, streamline_collate_fn


class StepsLR(Protocol):
    """Class for pylance to recognize the scheduler type"""
    def step(self, *args, **kwargs) -> None:
        ...

def _get_stratified_epoch_sampler(train_dataset: StreamlineDataset,
                                val_dataset: StreamlineDataset,
                                sampling_percentage_train: float,
                                sampling_percentage_val: float,
                                min_samples_per_class: int,
                                seed: int,
                                shuffle: bool,
                                verbose: bool) -> tuple[StratifiedEpochSampler, StratifiedEpochSampler]:

    train_sampler = StratifiedEpochSampler(dataset=train_dataset,
                                sampling_percentage=sampling_percentage_train,
                                min_samples_per_class=min_samples_per_class,
                                seed=seed,
                                shuffle=shuffle,
                                verbose=verbose)

    val_sampler = StratifiedEpochSampler(dataset=val_dataset,
                                sampling_percentage=sampling_percentage_val,
                                min_samples_per_class=min_samples_per_class,
                                seed=seed,
                                shuffle=shuffle,
                                verbose=verbose)    

    return train_sampler, val_sampler

def _get_loader(train_dir: str, 
                val_dir:str,
                batch_size: int,
                num_workers: int,
                shuffle: bool,
                sampling_percentage_train: float,
                sampling_percentage_val: float,
                max_streamlines: Optional[int],
                min_streamlines: int,
                seed: int,
                verbose: bool
                ) -> tuple[DataLoader, DataLoader, StratifiedEpochSampler, StratifiedEpochSampler]:
    """
    Returns the train and validation data loaders.
    """

    hdf5_train_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    hdf5_val_paths = [os.path.join(val_dir, f) for f in os.listdir(val_dir)]

    train_dataset = StreamlineDataset(hdf5_train_paths, 
                                    sampling_percentage=sampling_percentage_train,
                                    min_streamlines=min_streamlines,
                                    max_streamlines=max_streamlines,
                                    seed=seed,
                                    verbose=verbose)
    
    val_dataset = StreamlineDataset(hdf5_val_paths, 
                                    sampling_percentage=sampling_percentage_val, 
                                    min_streamlines=min_streamlines, 
                                    max_streamlines=max_streamlines, 
                                    seed=seed, 
                                    verbose=verbose)

    train_sampler, val_sampler = _get_stratified_epoch_sampler(train_dataset=train_dataset,
                                                                val_dataset=val_dataset,
                                                                sampling_percentage_train=sampling_percentage_train,
                                                                sampling_percentage_val=sampling_percentage_val,
                                                                min_samples_per_class=min_streamlines,
                                                                seed=seed,
                                                                shuffle=shuffle,
                                                                verbose=verbose
                                                            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=streamline_collate_fn,
        persistent_workers=False,
        # prefetch_factor=2 SERÁ NECESARIO?
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=streamline_collate_fn,
        persistent_workers=False,
        # prefetch_factor=2 SERÁ NECESARIO?
    )

    return train_loader, val_loader, train_sampler, val_sampler

def _get_encoder(encoder_type: str = "transformer",
                input_dim: int = 5,
                model_dim: int = 128,
                dim_feedforward: int = 512,
                num_heads: int = 8,
                num_layers: int = 4,
                dropout: float = 0.1,
                num_classes: int = 32,
                pooling_strategy: str = "mean",
                positional_encoding: str = "sinusoidal",
                bidirectional: bool = True
                ) -> nn.Module:

    if encoder_type == "transformer":
        return TransformerEncoder(input_dim=input_dim,
                                model_dim=model_dim,
                                dim_feedforward=dim_feedforward,
                                num_heads=num_heads,
                                num_layers=num_layers,
                                dropout=dropout,
                                num_classes=num_classes,
                                pooling_strategy=pooling_strategy,
                                positional_encoding=positional_encoding
                            )
    
    elif encoder_type == "lstm":
        return LSTMEncoder(input_dim=input_dim,
                        hidden_dim=model_dim,
                        num_layers=num_layers,
                        bidirectional=bidirectional,
                        dropout=dropout,
                        num_classes=num_classes,
                        pooling_strategy=pooling_strategy
                    )
    else:
        raise ValueError(f"Encoder type must be one of the following options: ['transformer', 'lstm'], got {encoder_type}")

def _parse_args(config):

    config = GlobalConfiguration()

    parser = argparse.ArgumentParser(description="Streamline classification training arguments.")

    parser.add_argument('--experiment_name', type=str, required=True,
                        help="Name of this experiment")

    parser.add_argument('--save_dir', type=str, default="checkpoints",
                        help="Directory to save the model checkpoints and logs. If not provided, defaults to './experiments/{experiment_name}'")

    parser.add_argument("--experiment_save_dir", type=str, default="results/training_experiments.csv",
                        help="Directory to save the model arguments used for the experiment.")

    parser.add_argument('--experiment_description', type=str, default=None,
                        help="Description or motive for the experiment")

    parser.add_argument('--train_dir', type=str, default=config.trainLoader_path,
                        help="Directory path of the training data")

    parser.add_argument('--val_dir', type=str, default=config.valLoader_path,
                        help="Directory path of the validation data")

    parser.add_argument('--encoder_type', type=str, default=config.encoder_type,
                        help="Name of the encoder it is going to be used")

    parser.add_argument('--loss_type', type=str, default=config.loss_fn_type, choices=['ce', 'focal'],
                        help="Name of the loss function to be used")

    parser.add_argument('--input_dim', type=int, default=config.input_dim,
                        help="Value of the input dimension")

    parser.add_argument('--batch_size', type=int, default=config.batch_size,
                        help="Batch size for training and validation")

    parser.add_argument('--num_classes', type=int, default=config.num_classes,
                        help="Number of classes to predict")

    parser.add_argument('--model_dim', type=int, default=config.model_dim,
                        help="Dimension of the model embeddings")

    parser.add_argument('--feedfoward_dim', type=int, default=config.feedforward_dim,
                        help="Dimension of the feedforward layers in the encoder")

    parser.add_argument('--num_heads', type=int, default=config.num_heads,
                        help="Num heads of the Transformer encoder")

    parser.add_argument('--num_layers', type=int, default=config.num_layers,
                        help="Number of layers of the encoder")

    parser.add_argument('--dropout', type=float, default=config.dropout,
                        help="Probability of dropout")

    parser.add_argument('--pooling_strategy', type=str, default=config.pooling_strategy, choices=['cls', 'mean', 'max'],
                        help="Pooling strategy for the encoder")

    parser.add_argument('--positional_encoding', type=str, default=config.positional_encoding,
                        help="Postional encoding to use in the Transformer encoder. Available values: ['sinusoidal']")

    parser.add_argument('--use_amp', action='store_false',
                        help="Activates or deactivates the mixed precision during training. Default True")

    parser.add_argument('--learning_rate', type=float, default=config.learning_rate,
                        help="Learning rate value")

    parser.add_argument('--weight_decay', type=float, default=config.weight_decay,
                        help="Wieght decay value")

    parser.add_argument('--num_epochs', type=int, default=config.num_epochs,
                        help="Number of epochs during training")

    parser.add_argument('--patience', type=int, default=config.patience,
                        help="Patience epochs before early stopping")

    parser.add_argument('--warmup_steps', type=int, default=config.warmup_steps,
                        help="Warmup steps for the learning rate")

    parser.add_argument('--accumulation-steps', type=int, default=config.accumulation_steps,
                        help="Number of steps for accumulating gradients")

    parser.add_argument('--no_bidirectional', action="store_false",
                        help="Deactivates LSTM bidertionality")

    parser.add_argument('--validate_every', type=int, default=config.validate_every,
                        help="Number of epochs before doing a validation epoch")    

    parser.add_argument('--num_workers', type=int, default=config.num_workers,
                        help="Number of workers for the dataloaders")

    parser.add_argument('--sampling_percentage_train', type=float, default=config.sampling_percentage_train,
                        help="Percentage of streamlines to sample from the training dataset")

    parser.add_argument('--sampling_percentage_val', type=float, default=config.sampling_percentage_val,
                        help="Percentage of streamlines to sample from the validation dataset")

    parser.add_argument('--min_streamlines', type=int, default=config.min_streamlines,
                        help="Minimum number of streamlines per tract")

    parser.add_argument('--max_streamlines', type=int, default=config.max_streamlines,
                        help="Maximum number of streamlines per tract")

    parser.add_argument('--seed', type=int, default=config.seed,
                        help="Random seed for reproducibility")

    parser.add_argument('--verbose', action='store_true',
                        help="Activates verbose mode for detailed logging")

    return parser.parse_args()

def _log_experiment(args: argparse.Namespace, csv_path: str = "experiments_log.csv") -> None:
    """
    Appends the current experiment's hyperparameters (all argparse args) as a
    new row in a shared CSV file. Creates the file with a header if it
    does not exist yet. If new args are added later, the header is extended
    automatically and old rows are backfilled with empty values.
    """
    args_dict = vars(args).copy()
    args_dict["timestamp"] = datetime.now().isoformat(timespec="seconds")

    file_exists = os.path.isfile(csv_path)

    if file_exists:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
        # Merge fieldnames: keep existing order, append any new ones at the end
        new_fields = [k for k in args_dict.keys() if k not in existing_header]
        fieldnames = existing_header + new_fields

        if new_fields:
            # Header changed: rewrite the whole file with the expanded header
            with open(csv_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    else:
        fieldnames = list(args_dict.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(args_dict)

    print(f"Experiment hyperparameters logged to {csv_path}")

def _create_hparams(args: argparse.Namespace) -> dict:
    """
    Creates a dictionary of hyperparameters from the argparse Namespace.
    """
    model_hparams = {
        'encoder_type': args.encoder_type,
        'input_dim': args.input_dim,
        'model_dim': args.model_dim,
        'dim_feedforward': args.feedfoward_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'num_classes': args.num_classes,
        'pooling_strategy': args.pooling_strategy,
        'positional_encoding': args.positional_encoding,
    }

    training_hparams = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'accumulation_steps': args.accumulation_steps,
        'warmup_steps': args.warmup_steps,
        'num_epochs': args.num_epochs,
        'patience': args.patience,
        'validate_every': args.validate_every,
        'use_amp': args.use_amp,
        'loss_type': args.loss_type,
        'seed': args.seed,
    }

    hparams = {**model_hparams, **training_hparams}

    return hparams

