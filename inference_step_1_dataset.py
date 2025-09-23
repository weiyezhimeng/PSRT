import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
import pandas as pd
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser()
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

def generate_response(model, tokenizer, prompt_embeds, user_input, device="cuda"):
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
    chat_template = [
        {"role": "user", "content": user_input},
        ]
    prompt_text = tokenizer.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
    # Tokenize the input (removing the first token as in training)
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
    
    # Generate response
    outputs = model.generate(
        inputs_embeds=combined_embeds,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Decode the generated tokens (skip the input part)
    generated_ids = outputs[:, :]
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    generated_len = generated_ids.shape[1]
    return response, generated_len

def main(dataset_path, output_json_path, model_name_or_path, prompt_embeds_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    model.eval()

    # Load prompt embeddings
    prompt_embeds = load_prompt_embeddings(prompt_embeds_path)
    
    ext = os.path.splitext(dataset_path)[1].lower()
    data_list = pd.read_csv(dataset_path) if ext == '.csv' else pd.read_json(dataset_path)
    
    results = []
    for index, item in tqdm(data_list.iterrows(), total=len(data_list), desc="processing"):
        response, response_len = generate_response(model, tokenizer, prompt_embeds, item['prompt'], device)
        print("\nResponse:", response, "\n")
        results_item = {
            "id": index,
            "prompt": item["prompt"],
            "response": response,
            "response_len": response_len,
        }

        if "label" in item:
            results_item["label"] = item["label"]

        results.append(results_item)
        
    
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

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    save_path = os.path.join(save_path_name, dataset_name, model_name)
    os.makedirs(save_path, exist_ok=True)
    final_save_path = os.path.join(save_path, f'{model_name}_{PROMPT_LENGTH}.json')

    model_name_or_path = f"{model_prefix}/{model_name}"
    prompt_embeds_path = f"{prompt_prefix}/{model_name}/prompt_embeds_final_{PROMPT_LENGTH}.pt"
    
    main(dataset_path, final_save_path, model_name_or_path, prompt_embeds_path, device)