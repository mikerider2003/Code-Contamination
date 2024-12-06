import os
import json
from datasets import load_dataset
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np

def convert_huggingface_data_to_list_dic(dataset):
    dataset = dataset["test"]
    all_data = []
    for task in dataset:
        all_data.append(task)

    return all_data

def generate_code(data, output_filename="output.json"):
    # Load the model
    model_name = "EleutherAI/pythia-2.8b"  # Example model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    with tqdm(total=len(data), desc="Generating Code") as pbar:
        for i in range(len(data)):
            prompt = data[i]['prompt']

            text = prompt
    
            input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
            input_ids = input_ids.to(model.device)
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
            loss, logits = outputs[:2]

            # Greedy decoding
            generated_ids = torch.argmax(logits, dim=-1)

            # Decode the generated IDs
            response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # mink and mink++
            input_ids = input_ids[0][1:].unsqueeze(-1)
            probs = F.softmax(logits[0, :-1], dim=-1)
            log_probs = F.log_softmax(logits[0, :-1], dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
            mu = (probs * log_probs).sum(-1)
            sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

            ## mink++
            mink_plus_scores = {}
            mink_plus = (token_log_probs - mu) / sigma.sqrt()
            for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                k_length = int(len(mink_plus) * ratio)
                topk = np.sort(mink_plus.cpu())[:k_length]
                mink_plus_scores[f'mink++_{ratio}'] = np.mean(topk).item()

            data[i]['generated_code'] = response
            data[i]['mkpp'] = mink_plus_scores

            print(data[i])
            save_data(new_data=data[i], filename=output_filename)

            pbar.update(1)

def save_data(new_data, filename='output.json'):
    # Check if file exists
    if os.path.exists(filename):
        # Load existing data
        with open(filename, 'r') as file:
            data = json.load(file)
    else:
        # If file does not exist, start with an empty list
        data = []

    # Append new data
    data.append(new_data)

    # Write updated data back to the JSON file
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    # Load dataset
    ds = load_dataset("openai/openai_humaneval")
    data_converted = convert_huggingface_data_to_list_dic(ds)

    # Generate Code
    generate_code(data=data_converted, output_filename="OpenAI_HumanEval_test.json")
