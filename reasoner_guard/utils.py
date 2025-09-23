import torch
import torch.distributed as dist
import random
import numpy as np
import os
import argparse

def parse_dtype(dtype_str):
    if dtype_str == 'bfloat16':
        return torch.bfloat16
    elif dtype_str == 'float32':
        return torch.float32
    else:
        raise argparse.ArgumentTypeError(f"Unsupported dtype: {dtype_str}. Choose from 'float16', 'float32'.")

def set_pad_token_id(tokenizer):
    if tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
        # print("hahahahha")
    elif tokenizer.eos_token_id is not None:
        pad_token_id = tokenizer.eos_token_id
    else:
        # define it by yourself
        pad_token_id = 11
    return pad_token_id

# ======================== Utilities ========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.distributed = True
    else:
        args.rank = 0
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.distributed = False

# ======================== Preprocessing ========================
def preprocess_function(ex, tokenizer):
    prompt_text = ex["question"]
    input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[:, :].squeeze(0).tolist()
    label_ids = tokenizer(ex["response"], return_tensors="pt", add_special_tokens=False).input_ids[:, :].squeeze(0).tolist()
    return {"input_ids": input_ids, "labels": label_ids, "len_input_ids": len(input_ids), "len_labels": len(label_ids)}

# ======================== Initialize Prompt ========================
def initialize_prompt_embeddings(reasonings, tokenizer, model, prompt_length, device, batch_size=32):
    with torch.no_grad():
        embed_layer = model.get_input_embeddings()
        total = 0
        prompt_tensor = torch.zeros(prompt_length, embed_layer.embedding_dim, device=device)
        
        for i in range(0, len(reasonings), batch_size):
            batch_texts = reasonings[i:i+batch_size]
            batch_embeds = []
            for text in batch_texts:
                ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :].to(device)
                embeds = embed_layer(ids).squeeze(0)
                if embeds.size(0) >= prompt_length:
                    selected = embeds[:prompt_length]
                else:
                    pad_len = prompt_length - embeds.size(0)
                    pad = torch.randn(pad_len, embeds.size(1), device=device)
                    selected = torch.cat([embeds, pad], dim=0)
                batch_embeds.append(selected)
                del ids, embeds
            batch_tensor = torch.stack(batch_embeds, dim=0)
            prompt_tensor += torch.sum(batch_tensor, dim=0)
            total += len(batch_texts)
            del batch_embeds, batch_tensor
            torch.cuda.empty_cache()
        prompt_tensor /= total
    print(prompt_tensor, flush=True)
    return prompt_tensor

def initialize_prompt_embeddings_old(reasonings, tokenizer, model, prompt_length, device):
    with torch.no_grad():
        embed_layer = model.get_input_embeddings()
        all_embeds = []
        for text in reasonings:
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :].to(device)
            embeds = embed_layer(ids).squeeze(0)
            if embeds.size(0) >= prompt_length:
                selected = embeds[:prompt_length]
            else:
                pad_len = prompt_length - embeds.size(0)
                pad = torch.zeros(pad_len, embeds.size(1), device=device)
                selected = torch.cat([embeds, pad], dim=0)
            all_embeds.append(selected)

        prompt_tensor = torch.mean(torch.stack(all_embeds, dim=0), dim=0)  # shape: [prompt_length, hidden_size]
    return prompt_tensor

def initialize_prompt_embeddings_from_pretrained(prompt_path, device):
    return torch.load(prompt_path).to(device)

def initialize_prompt_embeddings_random(model, prompt_length, device):
    embed_layer = model.get_input_embeddings()
    hidden_size = embed_layer.embedding_dim 
    prompt_tensor = torch.randn(prompt_length, hidden_size, device=device)

    return prompt_tensor