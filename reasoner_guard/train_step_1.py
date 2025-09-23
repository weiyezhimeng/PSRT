import os
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
import gc
from utils import *
from training_part import *
from torch.amp import autocast
from torch.amp import GradScaler
import swanlab

# ======================== Argument Parsing ========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--train_type", type=str, default="bfloat16")
    parser.add_argument("--prompt_length", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--max_total_length", type=int, default=2048)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--gradient_checkpointing_enable", action='store_true')
    parser.add_argument("--ablation_init", action='store_true')
    return parser.parse_args()

# ======================== Main Training ========================
def main():
    args = parse_args()
    init_distributed_mode(args)
    set_seed(args.seed)
    if not args.distributed or args.local_rank == 0:
        swanlab.init(project=os.path.basename(args.model_name_or_path), config=args)
        swanlab.config.update(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
    train_type = parse_dtype(args.train_type)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=train_type).to(args.device)
    if args.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()
    pad_token_id = set_pad_token_id(tokenizer)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    if args.dataset_path.endswith('.json'):
        dataset = load_dataset("json", data_files=args.dataset_path)["train"]
    elif args.dataset_path.endswith('.parquet'):
        dataset = load_dataset("parquet", data_files=args.dataset_path)["train"]
    else:
        raise ValueError("Unsupported file format. Only .json and .parquet files are supported.")
    reasonings = dataset["reasoning"]

    # ======================== set prompt embedding ========================
    if args.ablation_init:
        prompt_tensor = initialize_prompt_embeddings_random(model, args.prompt_length, args.device)
    else:
        prompt_tensor = initialize_prompt_embeddings(reasonings, tokenizer, model, args.prompt_length, args.device)
    
    prompt_module = PromptEmbeddingModule(prompt_tensor).to(args.device)
    if args.distributed:
        prompt_module = DDP(prompt_module, device_ids=[args.local_rank])
    # ======================== set prompt embedding ========================

    optimizer = torch.optim.AdamW(prompt_module.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    MAX_TOTAL_LENGTH = args.max_total_length 
    prompt_len = args.prompt_length

    def filter_long_sequences(example):
        total_len = len(example['input_ids']) + prompt_len + len(example['labels'])
        return total_len <= MAX_TOTAL_LENGTH

    tokenized_dataset = dataset.map(lambda ex: preprocess_function(ex, tokenizer), remove_columns=dataset.column_names)
    filtered_dataset = tokenized_dataset.filter(filter_long_sequences)
    balanced_dataset = filtered_dataset

    print(len(balanced_dataset), flush=True)

    sampler = DistributedSampler(balanced_dataset) if args.distributed else None
    dataloader = DataLoader(balanced_dataset, sampler=sampler, batch_size=args.per_device_batch_size, collate_fn = lambda batch: {
            "input_ids": torch.nn.utils.rnn.pad_sequence([torch.tensor(b["input_ids"], dtype=torch.long) for b in batch], batch_first=True, padding_value=pad_token_id),
            "labels": torch.nn.utils.rnn.pad_sequence([torch.tensor(b["labels"], dtype=torch.long) for b in batch], batch_first=True, padding_value=-100),
            "labels_embedding": torch.nn.utils.rnn.pad_sequence([torch.tensor(b["labels"], dtype=torch.long) for b in batch], batch_first=True, padding_value=1),
            "len_input_ids": [b["len_input_ids"] for b in batch],
            "len_labels": [b["len_labels"] for b in batch],
        }
    )

    total_steps = (len(dataloader) // args.gradient_accumulation_steps) * args.max_epochs
    scheduler = get_scheduler(args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=int(total_steps*args.warmup_ratio), num_training_steps=total_steps)

    model.train()
    global_step = 0
    
    pad_token_embed = model.get_input_embeddings()(torch.tensor([pad_token_id], device=args.device))  # (1, hidden_size)
    scaler = GradScaler(enabled=(train_type == torch.bfloat16))
    for epoch in range(args.max_epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            labels_embedding = batch["labels_embedding"].to(args.device)
            len_input_ids = batch["len_input_ids"]
            len_labels =  batch["len_labels"]

            batch_size = input_ids.size(0)
            pe = prompt_module.module.prompt_embeds if args.distributed else prompt_module.prompt_embeds
            prompt_len = pe.size(0)

            with torch.no_grad():
                inputs_embeds = model.get_input_embeddings()(input_ids)
                labels_embeds = model.get_input_embeddings()(labels_embedding)

            seq_lengths = [l_i + prompt_len + l_l for l_i, l_l in zip(len_input_ids, len_labels)]
            max_len = max(seq_lengths)

            full_inputs = pad_token_embed.unsqueeze(0).expand(batch_size, max_len, -1).clone().to(train_type)
            full_labels = torch.full((batch_size, max_len), -100, device=args.device, dtype=torch.long)
            full_attention_mask = torch.zeros((batch_size, max_len), device=args.device, dtype=torch.long)

            for i in range(batch_size):
                input_end = len_input_ids[i]
                prompt_end = input_end + prompt_len
                label_end = prompt_end + len_labels[i]

                full_inputs[i, :input_end] = inputs_embeds[i, :len_input_ids[i]]
                full_inputs[i, input_end:prompt_end] = pe
                full_inputs[i, prompt_end:label_end] = labels_embeds[i, :len_labels[i]]
                
                full_labels[i, prompt_end:label_end] = labels[i, :len_labels[i]]

                full_attention_mask[i, :label_end] = 1
            
            with autocast(device_type="cuda", dtype=torch.bfloat16): 
                outputs = model(inputs_embeds=full_inputs, labels=full_labels, attention_mask=full_attention_mask)
                loss = outputs.loss / args.gradient_accumulation_steps
        
            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0 and (not args.distributed or args.local_rank == 0):
                    print(f"Epoch {epoch} Step {global_step} Loss {loss.item() * args.gradient_accumulation_steps:.4f}")
                    current_lr = optimizer.param_groups[0]["lr"]
                    swanlab.log({
                        "train_loss": loss.item() * args.gradient_accumulation_steps,
                        "learning_rate": current_lr,
                        "epoch": epoch,
                        "global_step": global_step,
                    })

                if global_step % args.save_steps == 0 and (not args.distributed or args.local_rank == 0):
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    pe_to_save = prompt_module.module.prompt_embeds.detach().cpu() if args.distributed else prompt_module.prompt_embeds.detach().cpu()
                    torch.save(pe_to_save, os.path.join(save_path, "prompt_embeds.pt"))
            
            del outputs, loss
            del input_ids, labels
            del inputs_embeds, labels_embeds
            del full_inputs, full_attention_mask
            del full_labels
            gc.collect()
            torch.cuda.empty_cache()

    if not args.distributed or args.local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, os.path.basename(args.model_name_or_path)), exist_ok=True)
        final_pe = prompt_module.module.prompt_embeds.detach().cpu() if args.distributed else prompt_module.prompt_embeds.detach().cpu()
        torch.save(final_pe, os.path.join(args.output_dir, os.path.basename(args.model_name_or_path), f"prompt_embeds_final_{args.prompt_length}.pt"))
        swanlab.finish()

if __name__ == "__main__":
    main()