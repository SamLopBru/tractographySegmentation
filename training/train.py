import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from contextlib import nullcontext
from sklearn.metrics import f1_score
import time
import os
import sys
import gc
import argparse

from encoder import TransformerEncoder
from losses import _make_loss
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils.dataloader import StratifiedEpochSampler
from utils.config import GlobalConfiguration


def train_epoch(model: nn.Module,
                optimizer: torch.optim.Optimizer,
                dataloader: DataLoader,
                device: torch.device,
                criterion: nn.Module,
                scaler: GradScaler,
                accumulation_steps: int,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                use_amp: bool,
                log_interval: int = 100):
    
    model.train()
    
    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0.0, device=device)
    total_samples = 0
    total_grad_norm = 0.0
    grad_norm_count = 0

    optimizer.zero_grad()

    amp_ctx = autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()

    num_batches = len(dataloader)

    for batch_step, (streamlines, lengths, labels) in enumerate(dataloader):
        streamlines = streamlines.to(device) 
        lengths = lengths.to(device)
        labels = labels.to(device)

        with amp_ctx:
            logits = model(streamlines, lengths=lengths)
            loss = criterion(logits, labels) / accumulation_steps
        
        scaler.scale(loss).backward()

        if (batch_step + 1) % accumulation_steps == 0 or (batch_step + 1) == num_batches:

            scaler.unscale_(optimizer)

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isfinite(grad_norm):
                total_grad_norm += grad_norm.item()
                grad_norm_count += 1

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                scheduler.step()

            else:
                print(f"WARNING: skipping batch {batch_step + 1} due to inf/NaN gradients")
                optimizer.zero_grad()
                continue

            with torch.no_grad():
                total_loss += loss.detach() * labels.size(0)
                predictions = logits.argmax(dim=1)
                total_correct += (predictions == labels).sum() # esto lo podría quitar??
                total_samples += labels.size(0)

        if (batch_step + 1) % log_interval == 0:
            avg_loss = total_loss.item() / total_samples
            accuracy = 100.0 * total_correct.item() / total_samples
            print(f"    Batch {batch_step + 1} / {len(dataloader)} | "
                f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%"
            )

    return {
    'loss': total_loss.item() / total_samples,
    'accuracy': 100.0 * total_correct.item() / total_samples,
    'grad_norm': total_grad_norm / max(1, grad_norm_count)
    }

@torch.no_grad
def validate_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_labels = []
    all_preds = []

    amp_ctx = autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()

    # PORQUE NON_BLOCKING?
    for streamlines, lengths, labels in dataloader:
        streamlines = streamlines.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with amp_ctx:
            logits = model(streamlines, lengths)
            loss = loss_fn(logits, labels)

        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        # PORQUE PASARLO A CPU??
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predictions.cpu().numpy())

    return{
        'loss': total_loss / total_samples,
        'accuracy': 100.0 * total_correct / total_samples,
        'macro_f1': f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    }
    
def train_loop(model: nn.Module,
            criterion: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            train_sampler: StratifiedEpochSampler,
            val_sampler: StratifiedEpochSampler,
            device: torch.device,
            use_amp: bool,
            optimizer: torch.optim.Optimizer,
            # learning_rate: float,
            # weight_decay: float,
            num_epochs: int,
            patience: int,
            accumulation_steps: int,
            warmup_steps: int,
            scaler: GradScaler,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
            save_dir: str,
            validate_every: int = 2,
            verbose: bool = False):

    model = model.to(device)

    if use_amp and device.type != 'cuda':
        raise ValueError("AMP requires CUDA device, turn use_amp to True")
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'epoch_time': []  # Track time per epoch
    }

    best_val_f1 = 0.0
    patience_counter = 0
    start_epoch = 1

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        
        start_time = time.time()
        train_sampler.set_epoch(epoch)

        train_metrics = train_epoch(model=model, optimizer=optimizer, dataloader=train_loader,
                                    device=device, criterion=criterion, scaler=scaler,
                                    accumulation_steps=accumulation_steps, scheduler=scheduler,
                                    use_amp=use_amp)
        
        if verbose: 
            print(f"\n Train loss: {train_metrics['loss']:.4} | Train Acc: {train_metrics['accuracy']:.2}%")

        should_validate = (epoch % validate_every == 0) or (epoch == num_epochs)

        if should_validate:
            val_sampler.set_epoch(epoch)

            val_metrics = validate_epoch(model=model, loss_fn=criterion, dataloader=val_loader,
                                        device=device, use_amp=use_amp)
        else:
            # Placeholder metrics when skipping validation
            val_metrics = {'loss': history['val_loss'][-1] if history['val_loss'] else 0, 
                        'accuracy': history['val_acc'][-1] if history['val_acc'] else 0,
                        'macro_f1': history['val_f1'][-1] if history['val_f1'] else 0}
        
        epoch_time = time.time() - start_time

        # For pylance compliance    
        if val_metrics is None:
                raise ValueError("Validation metric dictionary has not been created.")
    
        if verbose:
            if not should_validate:
                print("Skipping validation epoch")

            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}% | Val F1: {val_metrics['macro_f1']:.2f}% | Time: {epoch_time:.1f}s")

        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['macro_f1'])
        history['epoch_time'].append(epoch_time)

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'val_macro_f1': best_val_f1,
                'history': history,
                # 'params': hparams
            }

            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))

            if verbose:
                print(f"    Saved new best model (Validation F1: {best_val_f1:.2f})%, Validation Acc: {val_metrics['accuracy']:.2f}%")
        
        else:
            patience_counter +=1
            if patience_counter >= patience:
                print(f"EARLY STOPPING TRIGGERED AFER {epoch} EPOCHS")
                break
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'warmup_cosine_scheduler_state': warmup_cosine_scheduler.state_dict(),
            # 'plateau_scheduler_state': plateau_scheduler.state_dict() if plateau_scheduler is not None else None,
            'scaler_state': scaler.state_dict() if scaler is not None else None,
            'best_val_f1': best_val_f1,
            'patience_counter': patience_counter,
            'history': history
        }, os.path.join(save_dir, 'latest_checkpoint.pt'))

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if verbose:

        print(f"\n{'='*60}")
        print(f"Training complete! Best validation Macro F1: {best_val_f1:.2f}%")
        print('='*60)
    
    return history


def main():

    config = GlobalConfiguration

    parser = argparse.ArgumentParser(description="Streamline classification training arguments.")


    parser.add_argument('--experiment-name', type=str, required=True,
                        help="Name of this experiment")

    parser.add_argument('--experiment-description', type=str, default=None,
                        help="Description or motive for the experiment")

    parser.add_argument('--train-dir', type=str, default=config.trainLoader_path,
                        help="Directoyy path of the training data")

    parser.add_argument('--val-dir', type=str, default=config.valLoader_path,
                        help="Directory path of the validation data")

    parser.add_argument('--encoder-type', type=str, default=config.encoder_type,
                        help="Name of the encoder it is going to be used")

    parser.add_argument('--loss_type', str=str, default=config.loss_fn_type, choices=['ce', 'focal'],
                        help="Name of the loss function to be used")
    
    parser.add_argument('--input_dim', type=int, default=config.input_dim,
                        help="Value of the input dimension")

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

    parser.add_argument('--loss_fn', type=int, default=config.loss_fn_type,
                        help="Loss function to be used during training. Available values: ['ce', 'focal']")

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

    parser.add_argument('--validate_every', type=int, default=config.validate_every,
                        help="Number of epochs before doing a validation epoch")    

    args = parser.parse_args()

    model = TransformerEncoder(input_dim=args.input_dim, 
                            model_dim=args.model_dim,
                            dim_feedforward=args.feedfoward_dim,
                            num_heads=args.num_heads,
                            num_layers=args.num_layers,
                            dropout=args.dropout,
                            num_classes=args.num_classes,
                            pooling_strategy=args.pooling_strategy,
                            positional_encoding=args.positional_encoding
                            )

    criterion = _make_loss(args.loss_type)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Learning rate warmup steps
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=args.learning_rate, total_iters=args.warmup_steps)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.num_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[warmup_scheduler,cosine_scheduler], milestones=[args.warmup_steps])


    





if __name__ == "__main__":
    pass

    
