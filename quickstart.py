"""
Single-GPU quick start for training ImprovedGPTModel with the Saw activation function.

Targets an effective batch size of 512:
    batch_size (32) * num_devices (1) * accumulation_steps (16) = 512

For multi-GPU or resuming from a checkpoint, see README.md.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from datasets import load_dataset
import tiktoken

from improved_model import ImprovedGPTModel
from improved_datautils import create_dataloader
from trainingutils import training_loop
from saw import Saw

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Build model with a custom activation ---
model = ImprovedGPTModel(GPT_CONFIG_124M, activation_fn=Saw).to(device)

# --- Dataset (TinyStories) ---
tokenizer = tiktoken.get_encoding("gpt2")

def tokenize_function(examples):
    eot = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    context_length = GPT_CONFIG_124M["context_length"]
    all_tokens = []
    for text in examples["text"]:
        all_tokens.extend(tokenizer.encode(text) + [eot])
    inputs, targets = [], []
    for i in range(0, len(all_tokens) - context_length, context_length):
        inputs.append(all_tokens[i : i + context_length])
        targets.append(all_tokens[i + 1 : i + context_length + 1])
    return {"input_ids": inputs, "target_ids": targets}

dataset = load_dataset("roneneldan/TinyStories", split="train")
tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized.set_format(type="torch", columns=["input_ids", "target_ids"])

split = tokenized.train_test_split(test_size=0.2, seed=1337)
train_dl = create_dataloader(split["train"], batch_size=32) # ensure the batch size here fits onto your GPU. You can increase it if you have a larger GPU, or decrease it if you have a smaller one. The effective batch size is batch_size * accumulation_steps, so you can adjust accumulation_steps accordingly to maintain an effective batch size of 512.
val_dl   = create_dataloader(split["test"],  batch_size=32)

optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)

# --- Train ---
# accumulation_steps * batch_size = 512  (32 * 16 = 512)
training_loop(
    start_epoch=0,
    total_epochs=5,
    start_batch_idx=0,
    train_dataloader=train_dl,
    val_dataloader=val_dl,
    model=model,
    optimizer=optimizer,
    accumulation_steps=16,
    device=device,
    checkpoint_root="Checkpoints",
    save_checkpoints=True,
    af_name="Saw",
    silent=False,
    multinode=False,
    batch_print_breakpoint=100,
    cfg=GPT_CONFIG_124M,
)
