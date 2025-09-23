import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
import pandas as pd
from tqdm import tqdm
import os
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--judge_mode', action='store_true')
    parser.add_argument("--PROMPT_LENGTH", type=int)
    parser.add_argument("--model_prefix", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--prompt_prefix", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--save_path_name", type=str)
    parser.add_argument("--device", type=str)
    return parser.parse_args()

def load_prompt_embeddings(prompt_path):
    """Load the trained prompt embeddings"""
    return torch.load(prompt_path)

def calculate_sequence_log_prob(model, embeds, target_ids):
    if target_ids.shape[1] == 0:
        return -float('inf')
        
    embedding_layer = model.get_input_embeddings()
    target_embeds = embedding_layer(target_ids.to(embeds.device))
    full_embeds = torch.cat([embeds, target_embeds], dim=1)
    
    with torch.no_grad():
        outputs = model(inputs_embeds=full_embeds)
    
    logits = outputs.logits
    prompt_len = embeds.shape[1]
    
    relevant_logits = logits[:, prompt_len - 1:-1, :]
    log_probs = F.log_softmax(relevant_logits, dim=-1)
    
    target_log_probs = torch.gather(log_probs, 2, target_ids.view(1, -1, 1)).squeeze(-1)
    return torch.sum(target_log_probs).item()

def generate_response(model, tokenizer, prompt_embeds, user_input, judge_mode, device="cuda"):
    """
    Generate response using the model with prompt embeddings
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt_embeds: The trained prompt embeddings tensor
        user_input: User's input text
        device: Device to run on
    """
    # Prepare the input
    if judge_mode:
        chat_template = [
            {"role": "user", "content": user_input},
        ]
    else:
        chat_template = [
            {"role": "user", "content": user_input},
        ]
    prompt_text = tokenizer.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
    # Tokenize the input (removing the first token as in training)
    # print(prompt_text)
    input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[:, :].to(device)
    # Get input embeddings
    with torch.no_grad():
        input_embeds = model.get_input_embeddings()(input_ids)
    
    # Prepare prompt embeddings (expand to batch size 1)
    prompt_embeds = prompt_embeds.unsqueeze(0).to(device)
    
    prompt_think_end = "</think>\n<answer>\n"
    prompt_think_end_ids = tokenizer(prompt_think_end, return_tensors="pt", add_special_tokens=False).input_ids[:, :].to(device)

    with torch.no_grad():
        prompt_think_end_embeds = model.get_input_embeddings()(prompt_think_end_ids)

    # Combine input and prompt embeddings
    combined_embeds = torch.cat([input_embeds, prompt_embeds, prompt_think_end_embeds], dim=1)
    # print(combined_embeds.shape)

    safe_ids = tokenizer.encode("<safe>\n", add_special_tokens=False, return_tensors="pt").to(device)
    # safe_ids_wrong = tokenizer.encode("<safe>\nI cannot", add_special_tokens=False, return_tensors="pt").to(device)
    unsafe_ids = tokenizer.encode("<unsafe>\n", add_special_tokens=False, return_tensors="pt").to(device)

    log_prob_safe = calculate_sequence_log_prob(model, combined_embeds, safe_ids)
    # log_prob_safe_wrong = calculate_sequence_log_prob(model, combined_embeds, safe_ids_wrong)
    log_prob_unsafe = calculate_sequence_log_prob(model, combined_embeds, unsafe_ids)

    log_probs_tensor = torch.tensor([log_prob_safe, log_prob_unsafe])
    normalized_probs = F.softmax(log_probs_tensor, dim=0)

    return normalized_probs[0].item(), normalized_probs[1].item()

def main(dataset_path, output_json_path, model_name_or_path, prompt_embeds_path, judge_mode, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    model.eval()

    # Load prompt embeddings
    prompt_embeds = load_prompt_embeddings(prompt_embeds_path)
    
    ext = os.path.splitext(dataset_path)[1].lower()
    data_list = pd.read_csv(dataset_path) if ext == '.csv' else pd.read_json(dataset_path)
    
    results = []
    for index, item in tqdm(data_list.iterrows(), total=len(data_list), desc="processing"):
        safe_prob, unsafe_prob = generate_response(model, tokenizer, prompt_embeds, item['prompt'], judge_mode, device)
        # print("\nResponse:", response, "\n")
        if safe_prob < unsafe_prob:
            label = 0
        else:
            label = 1
        print(item['prompt'], flush=True)
        print(safe_prob, unsafe_prob, flush=True)
        print(label, flush=True)
        results.append({
                "id": index,
                "prompt": item["prompt"],
                "safe_prob": safe_prob,
                "unsafe_prob": unsafe_prob,
                "label": label,
            })
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    args = parse_args()
    PROMPT_LENGTH = args.PROMPT_LENGTH
    model_prefix = args.model_prefix
    model_name = args.model_name
    prompt_prefix = args.prompt_prefix
    dataset_path = args.dataset_path
    save_path_name = args.save_path_name
    device = args.device
    judge_mode = args.judge_mode

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    if judge_mode:
        save_path = os.path.join("./result_step_2_judge_mode", dataset_name, model_name)
    else:
        save_path = os.path.join(save_path_name, dataset_name, model_name)
    os.makedirs(save_path, exist_ok=True)
    final_save_path = os.path.join(save_path, f'{model_name}_{PROMPT_LENGTH}.json')

    model_name_or_path = f"{model_prefix}/{model_name}"
    prompt_embeds_path = f"{prompt_prefix}/{model_name}/prompt_embeds_final_{PROMPT_LENGTH}.pt"
    
    main(dataset_path, final_save_path, model_name_or_path, prompt_embeds_path, judge_mode, device)