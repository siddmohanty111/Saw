import tiktoken
from tiktoken import Encoding
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

def _chunk_texts(examples, tokenizer : Encoding, max_length, stride):
    # Get the special end-of-text token ID for GPT-2 (usually 50256)
    eot_token = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    
    # 1. Tokenize a batch of texts and append the EOT token to separate documents
    tokenized_batch = [tokenizer.encode(text) + [eot_token] for text in examples['text']]
    
    # 2. Flatten all tokens into a single 1D stream
    all_tokens = []
    for doc in tokenized_batch:
        all_tokens.extend(doc)
        
    # 3. Create the overlapping sliding windows (chunks)
    input_ids = []
    target_ids = []
    
    # We slice max_length + 1 so we have both the input and the shifted target
    for i in range(0, len(all_tokens) - max_length, stride):
        chunk = all_tokens[i : i + max_length + 1]
        
        # Only keep chunks that are exactly the right size (drops the tiny remainder at the end)
        if len(chunk) == max_length + 1:
            input_ids.append(chunk[:-1])
            target_ids.append(chunk[1:])
            
    return {"input_ids": input_ids, "target_ids": target_ids}


def create_dataloader(hf_dataset, batch_size=4, drop_last=True, num_workers=0):
    
    # We don't need the GPTDataset class anymore because 
    # the dataset is already tokenized and formatted as tensors.
    
    sampler = DistributedSampler(hf_dataset) if torch.distributed.is_initialized() else None
    
    dataloader = DataLoader(
        hf_dataset, 
        batch_size=batch_size, 
        drop_last=drop_last, 
        num_workers=num_workers,
        pin_memory=True, # Helps speed up data transfer to GPU
        sampler=sampler
    )

    return dataloader