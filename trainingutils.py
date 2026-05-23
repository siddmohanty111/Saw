from threading import local

import torch

from .improved_model import ImprovedGPTModel
from .checkpointing import save_checkpoint

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn import DataParallel, CrossEntropyLoss, Linear
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

from typing import Dict, List

def make_checkpoint_path(root, af_name, additional=""):
    return f"{root}/{af_name}_checkpoint{additional}.pth"

def validate(model, val_dataloader, device, cfg):
    """Calculate average loss on validation data without training."""
    model.eval()  # Set to evaluation mode
    loss_fn = CrossEntropyLoss()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():  # Disable gradient computation
        for batch in val_dataloader:
            inputs = batch["input_ids"].to(device)
            targets = batch["target_ids"].to(device)
            
            # Forward pass only
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(inputs)
                loss = loss_fn(logits.view(-1, cfg["vocab_size"]), targets.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    model.train()  # Set back to training mode
    avg_loss = total_loss / num_batches
    return avg_loss

def training_loop(start_epoch : int, 
                  total_epochs : int,
                  start_batch_idx : int, 
                  train_dataloader : DataLoader, 
                  val_dataloader : DataLoader,
                  model : ImprovedGPTModel | DataParallel[ImprovedGPTModel] | DDP, 
                  optimizer : AdamW, 
                  accumulation_steps : int, 
                  device, 
                  checkpoint_root : str, 
                  save_checkpoints : bool,
                  af_name : str,
                  silent : bool,
                  multinode : bool,
                  batch_print_breakpoint : int, 
                  cfg,
                  history_train_losses : List[Dict] | None = None,
                  history_val_losses : List[Dict] | None = None) -> float:
    
    loss_fn = CrossEntropyLoss()
    if multinode:
        rank = int(os.environ.get("RANK", 0))
    else: 
        rank = int(os.environ.get("LOCAL_RANK", 0))

    val_loss = float('inf')

    for epoch in range(start_epoch, total_epochs):  # resume from saved epoch

        sampler = train_dataloader.sampler
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        # if resuming mid-epoch, need to skip the dataloder to the place that training failed
        data_iter = iter(train_dataloader)
        if start_batch_idx > 0: # if our batch number is non-zero at the start of this loop, that means training failed mid-epoch
            print(f"Fast-forwarding dataloader to batch {start_batch_idx}...")
            for _ in range(start_batch_idx):
                next(data_iter)
        
        # Continue (or start) training loop for this epoch as normal.
        for batch_idx, batch in enumerate(data_iter, start=start_batch_idx):
            
            # Inputs & Targets
            inputs : torch.Tensor = batch["input_ids"].to(device)
            targets : torch.Tensor = batch["target_ids"].to(device)

            # Forward Pass & Loss
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits : torch.Tensor = model(inputs)
                loss : torch.Tensor = loss_fn(logits.view(-1, cfg["vocab_size"]), targets.view(-1))
                loss = loss / accumulation_steps
            
            # Backward
            loss.backward()

            # Step
            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_dataloader)):

                # Gradient Clipping (model is overshooting on multiple GPUs)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 
                optimizer.step()
                optimizer.zero_grad()

            # --- CHECKPOINTING ---
            # Save every 500 batches, but not batch 0 (no training done yet)
            if save_checkpoints and batch_idx % 500 == 0 and batch_idx > 0:
                if rank == 0:
                    print(f"Saving checkpoint at Epoch {epoch}, Batch {batch_idx}...")
                    save_checkpoint(model, optimizer, epoch, batch_idx, loss.item(), make_checkpoint_path(checkpoint_root, af_name))
                dist.barrier()

            if batch_idx % batch_print_breakpoint == 0 and not silent and batch_idx > 0:
                if rank == 0:
                    current_loss = loss.item() * accumulation_steps
                    print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {current_loss:.4f}")
                    if history_train_losses is not None:
                        history_train_losses.append({
                            "epoch": epoch,
                            "batch": batch_idx,
                            "loss": current_loss
                        })
                # wait while rank 0 is printing stuff out and adding to the history before other ranks continue to avoid rank 0 slowly falling behind
                dist.barrier()
        
        # Validate BEFORE saving epoch checkpoint so dist.all_reduce never waits on disk I/O.
        # torch.save() on a new file can stall for minutes on Lustre/NFS under load,
        # which would cause all non-zero ranks to timeout at a barrier.
        val_loss = validate(model, val_dataloader, device, cfg)
        val_loss_tensor = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        val_loss = val_loss_tensor.item()

        if rank == 0:
            print(f"Epoch {epoch} | Validation Loss: {val_loss:.4f}")
            if history_val_losses is not None:
                history_val_losses.append({
                    "epoch": epoch,
                    "val_loss": val_loss
                })

        # Save epoch checkpoint after validation. Only rank 0 writes; no barrier is
        # needed here — other ranks simply move on to the next epoch (or exit the loop).
        # Losing this checkpoint on a crash is acceptable because mid-epoch 500-batch
        # checkpoints already cover recovery.
        final_epoch_checkpoint_path = make_checkpoint_path(checkpoint_root, af_name, additional=f"epoch_{epoch}")
        if save_checkpoints and rank == 0:
            save_checkpoint(model, optimizer, epoch, batch_idx, loss.item(), final_epoch_checkpoint_path) # type: ignore
            print(f"Epoch {epoch} complete. Final checkpoint for epoch {epoch} saved to {final_epoch_checkpoint_path}")

        # Reset start_batch for next epoch
        start_batch_idx = 0

    return val_loss