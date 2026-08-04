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
import pandas as pd

from losses import _make_loss
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils.dataloader import StratifiedEpochSampler
from utils.config import GlobalConfiguration
from utils.helpers import _get_loader, _get_encoder, _parse_args, _log_experiment, StepsLR


def train_epoch(model: nn.Module,
                optimizer: torch.optim.Optimizer,
                dataloader: DataLoader,
                device: torch.device,
                criterion: nn.Module,
                scaler: GradScaler,
                accumulation_steps: int,
                scheduler: StepsLR,
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

@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
):
    model.eval()

    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0.0, device=device)
    total_samples = 0

    all_labels = []
    all_preds = []

    amp_ctx = autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()

    for streamlines, lengths, labels in dataloader:
        streamlines = streamlines.to(device, non_blocking=True) # asynchronous host-to-device copy
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with amp_ctx:
            logits = model(streamlines, lengths)
            loss = loss_fn(logits, labels)

        total_loss += loss.detach() * labels.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum()
        total_samples += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predictions.cpu().numpy())

    return{
        'loss': (total_loss / total_samples).item(),
        'accuracy': 100.0 * (total_correct / total_samples).item(),
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
            learning_rate: float,
            weight_decay: float,
            num_epochs: int,
            patience: int,
            accumulation_steps: int,
            save_dir: str,
            warmup_steps: int,
            validate_every: int = 2,
            verbose: bool = False):

    model = model.to(device)

    if use_amp and device.type != 'cuda':
        raise ValueError("AMP requires CUDA device, turn use_amp to True")

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    # No applying weight decay to bias and normalization layers (https://arxiv.org/pdf/2305.17212)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=learning_rate)

    scaler = GradScaler(enabled=use_amp)

    steps_per_epoch = len(train_loader) // accumulation_steps
    total_training_steps = steps_per_epoch * num_epochs

    # Scheduler setup: Warmup + Cosine Annealing
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_training_steps - warmup_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[warmup_scheduler,cosine_scheduler], milestones=[warmup_steps])
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'epoch_time': []  # Track time per epoch
    }

    best_val_f1 = 0.0
    patience_counter = 0

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        
        start_time = time.time()
        train_sampler.set_epoch(epoch)

        train_metrics = train_epoch(model=model, optimizer=optimizer, dataloader=train_loader,
                                    device=device, criterion=criterion, scaler=scaler,
                                    accumulation_steps=accumulation_steps, scheduler=scheduler,
                                    use_amp=use_amp)
        
        if verbose: 
            print(f"\n Train loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")

        should_validate = ((epoch % validate_every == 0) or (epoch == num_epochs - 1) and (epoch != 0))

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

        break
    if verbose:

        print(f"\n{'='*60}")
        print(f"Training complete! Best validation Macro F1: {best_val_f1:.2f}%")
        print('='*60)
    
    return history


def main():

    config = GlobalConfiguration()
    args = _parse_args(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the loss function
    criterion = _make_loss(args.loss_type, device)

    # Initialize Encoder model
    model = _get_encoder(encoder_type=args.encoder_type, 
                        input_dim=args.input_dim,
                        model_dim=args.model_dim,
                        dim_feedforward=args.feedfoward_dim,
                        num_heads=args.num_heads,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        num_classes=args.num_classes,
                        pooling_strategy=args.pooling_strategy,
                        positional_encoding=args.positional_encoding)

    # Create data loaders and samplers
    train_loader, val_loader, train_sampler, val_sampler = _get_loader(train_dir=args.train_dir, 
                            val_dir=args.val_dir, 
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            sampling_percentage_train=args.sampling_percentage_train,
                            sampling_percentage_val=args.sampling_percentage_val,
                            min_streamlines=args.min_streamlines,
                            max_streamlines=args.max_streamlines,
                            shuffle=True,
                            seed=args.seed,
                            verbose=args.verbose)

    
    train_loop(model=model,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            val_sampler=val_sampler,
            num_epochs=args.num_epochs,
            device=device,
            use_amp=args.use_amp,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            accumulation_steps=args.accumulation_steps,
            warmup_steps=args.warmup_steps,
            save_dir=os.path.join(args.save_dir,args.experiment_name),
            validate_every=args.validate_every,
            verbose=args.verbose)



    _log_experiment(args, csv_path=args.experiment_save_dir)


if __name__ == "__main__":
    main()




    
