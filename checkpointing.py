import os
import torch
from torch.nn import DataParallel
from .improved_model import ImprovedGPTModel
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import glob

def save_checkpoint(model : ImprovedGPTModel | DataParallel[ImprovedGPTModel] | DDP, optimizer : AdamW, epoch, batch_idx, loss, filename="checkpoint.pth"):
    # Check if the model is wrapped in DataParallel
    # We want to save the 'raw' state_dict to keep checkpoints compatible
    if isinstance(model, DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, DDP):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    print("loaded model state dict for checkpointing")

    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': state_dict, # Always the raw weights
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    print(f"Saving checkpoint to {filename}...")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # --- Atomic Saving (disabled: os.replace/rename can hang on Lustre/NFS) ---
    # temp_filename = filename + ".tmp"
    # torch.save(checkpoint, temp_filename)
    # os.replace(temp_filename, filename)
    # --------------------------------------------------------------------------
    torch.save(checkpoint, filename)
    print(f"Checkpoint safely saved: {filename}")

    if "epoch" not in filename: 
        directory = os.path.dirname(filename)
        base_name = os.path.basename(filename)
        # Derive the prefix for this specific run (e.g. "ReLU_run1_checkpoint")
        # so we never touch checkpoints belonging to other concurrent jobs.
        own_prefix = base_name.replace(".pth", "")
        for old_chkpt in glob.glob(os.path.join(directory, f"{own_prefix}*.pth")):
            if old_chkpt != filename and "epoch" not in old_chkpt:
                try:
                    os.remove(old_chkpt)
                except OSError:
                    pass

def load_checkpoint(filename, model : ImprovedGPTModel | DataParallel[ImprovedGPTModel] | DDP, optimizer : AdamW):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'...")
        checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
        
        # Determine if we are loading INTO a DataParallel model or a regular one
        if isinstance(model, DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch']
        start_batch_idx = checkpoint.get('batch_idx', 0) + 1 
        print(f"Resuming from Epoch {start_epoch}, Batch {start_batch_idx}")
        return start_epoch, start_batch_idx
    else:
        print(f"No checkpoint found at '{filename}'. Starting from scratch.")
        return 0, 0