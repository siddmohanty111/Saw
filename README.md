# Saw — GPT-124M Training with Custom Activation Functions

This repository contains the modular training stack used for the experiments comparing the custom activation function Saw against ReLU in a GPT-2-scale language model trained on the TinyStories dataset.

---

## Repository Structure

```
Saw/
├── quickstart.py          # Single-GPU training script (start here)
├── improved_model.py      # GPT model definition (MultiHeadAttention, TransformerBlock, ImprovedGPTModel)
├── improved_datautils.py  # HuggingFace dataset tokenization and DataLoader creation
├── trainingutils.py       # Training loop utilities with gradient accumulation, DDP, and validation
├── checkpointing.py       # Checkpoint save/load utilities
└── saw.py                 # Programmatic definition of the Saw activation function
```

---

## Model Architecture

The model is a decoder-only GPT-2-scale transformer (`ImprovedGPTModel`) with the following design choices:

- **Flash Attention** via `F.scaled_dot_product_attention` with `is_causal=True` — replaces manual masking/softmax for efficiency on NVIDIA GPUs.
- **Pre-LayerNorm** residual blocks (norm is applied before attention and feed-forward, not after).
- **Pluggable activation functions** — the feed-forward sublayer accepts any `nn.Module` factory via `activation_fn`, defaulting to `nn.ReLU`.
- **Mixed precision** — the training loop uses `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`.
- **Gradient clipping** — `max_norm=1.0` applied before every optimizer step.

### Config to Replicate Paper Results

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,       # GPT-2 BPE vocabulary
    "context_length": 256,     # Sequence length
    "emb_dim": 768,            # Embedding / model dimension
    "n_heads": 12,             # Number of attention heads
    "n_layers": 12,            # Number of transformer blocks
    "drop_rate": 0.1,          # Dropout probability
    "qkv_bias": False          # No bias in Q/K/V projections
}
```

---

## Effective Batch Size

All results were produced with an **effective batch size of 512 tokens**. This is the product:

```
effective_batch_size = batch_size × num_devices × accumulation_steps = 512
```

How you achieve this depends on your hardware:

| Scenario | `batch_size` (per DataLoader) | `num_devices` | `accumulation_steps` |
|---|---|---|---|
| Supercomputer (e.g. 8× A100) | 64 | 8 | 1 |
| 4-GPU node | 64 | 4 | 2 |
| 2-GPU workstation | 64 | 2 | 4 |
| Single GPU (16 GB+) | 32 | 1 | 16 |
| Single GPU (8 GB) | 16 | 1 | 32 |
| CPU / low RAM | 4 | 1 | 128 |

> **Note:** `batch_size=512` in a single DataLoader call requires a large amount of RAM/VRAM and will likely crash your system if the code is not running in a supercomputing environment. Use gradient accumulation on consumer hardware.

---

## Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tiktoken datasets transformers
```

Python 3.10+ and PyTorch 2.0+ are required for Flash Attention (`F.scaled_dot_product_attention`) and `bfloat16` autocast support.

---

## Quick Start — Single GPU

> **Note:** Training the full 124M model on TinyStories on a single GPU will be very slow. A multi-GPU setup or HPC cluster is strongly recommended for actual experiments.

A self-contained single-GPU training script is provided in `quickstart.py`. Run it directly:

```bash
python quickstart.py
```

It trains `ImprovedGPTModel` with the `Saw` activation on the TinyStories dataset, targeting an effective batch size of 512 via gradient accumulation:

```
batch_size (32) × num_devices (1) × accumulation_steps (16) = 512
```

Many consumer devices will likely be able to achieve a higher batch size of 64, reducing accumulation steps to 8, if needed. 

---

## Multi-GPU (DDP) Training

Use `torchrun` to launch distributed training. The `training_loop` and `create_dataloader` functions automatically detect an initialized process group and use `DistributedSampler` and `DistributedDataParallel`.

```bash
# 4 GPUs on a single node
torchrun --nproc_per_node=4 your_train_script.py
```

In your script, initialize DDP before building the model:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

model = ImprovedGPTModel(GPT_CONFIG_124M, activation_fn=Saw).to(device)
model = DDP(model, device_ids=[local_rank])
```

For a 4-GPU setup targeting an effective batch size of 512:

```python
# batch_size=64 × 4 devices × accumulation_steps=2 = 512
train_dl = create_dataloader(train_dataset, batch_size=64, num_workers=4)
# pass accumulation_steps=2 to training_loop
```

---

## Checkpointing

Checkpoints are saved every 500 batches and at the end of every epoch by rank 0. To resume training:

```python
from checkpointing import load_checkpoint

model = ImprovedGPTModel(GPT_CONFIG_124M, activation_fn=Saw).to(device)
optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)

start_epoch, start_batch_idx = load_checkpoint("Checkpoints/Saw_checkpoint.pth", model, optimizer)
```

Pass `start_epoch` and `start_batch_idx` directly to `training_loop`. The loop will fast-forward the dataloader to the correct batch automatically.

---

## The Saw Activation Function

`Saw` is defined in `saw.py`. It is a continuous, piecewise-linear function with three regions:

```
x < left_break:                  out = x − 2·left_break   (slope +1)
left_break ≤ x < right_break:   out = −x                  (slope −1)
x ≥ right_break:                 out = x − 2·right_break  (slope +1)
```

Default breakpoints: `left_break = -1.0`, `right_break = 1.0`.

Pass `learnable=True` to make the breakpoints trainable `nn.Parameter`s:

```python
model = ImprovedGPTModel(GPT_CONFIG_124M, activation_fn=lambda: Saw(learnable=True))
```

---

## Swapping Activation Functions

Any `nn.Module` subclass can be plugged in as the feed-forward activation:

```python
# ReLU baseline
model = ImprovedGPTModel(GPT_CONFIG_124M, activation_fn=nn.ReLU)

# Saw
from saw import Saw
model = ImprovedGPTModel(GPT_CONFIG_124M, activation_fn=Saw)

# Learnable Saw
model = ImprovedGPTModel(GPT_CONFIG_124M, activation_fn=lambda: Saw(learnable=True))

# Any standard PyTorch activation
model = ImprovedGPTModel(GPT_CONFIG_124M, activation_fn=nn.GELU)
```

`activation_fn` must be a **zero-argument callable** (a class or a lambda) that returns an `nn.Module` instance — this is required so that each `TransformerBlock` gets its own independent instance.
