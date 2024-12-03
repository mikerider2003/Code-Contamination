import os
import json
from datasets import load_dataset
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def convert_huggingface_data_to_list_dic(dataset):
    dataset = dataset["test"]
    all_data = []
    for task in dataset:
        all_data.append(task)

    return all_data

def generate_code(data, output_filename="output.json"):
    # Load the model and tokenizer
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Example model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    with tqdm(total=len(data), desc="Generating Code") as pbar:
        for i in range(len(data)):

            # TODO: ADJUST Code bellow to work with meta-llama/Llama-3.1-8B-Instruct
            raise NotImplementedError
            
            prompt = data[i]['prompt']

            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You will get Python programming problem. You have to reply only with executable python code, no extra text."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # Generate the sequence and capture logits instead of only tokens
            with torch.no_grad():
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            # Separate the generated IDs and their corresponding logits (scores)
            generated_ids = outputs.sequences[:, model_inputs.input_ids.shape[-1]:]  # Only the generated portion
            logits = torch.stack(outputs.scores, dim=1)  # Logits for each generated token

            # Compute probabilities and log probabilities
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Create a mask for probs > 0
            mask = probs > 0

            # Compute products safely using the mask
            products = torch.where(mask, probs * log_probs, torch.zeros_like(probs))
            mu = products.sum(dim=-1)                 # Shape: [1, seq_len]

            # Compute expected squared log_probs safely
            products_squared = torch.where(mask, probs * log_probs.square(), torch.zeros_like(probs))
            expected_log_probs_squared = products_squared.sum(dim=-1)  # Shape: [1, seq_len]

            # Compute variance and ensure it's non-negative
            variance = expected_log_probs_squared - mu.square()
            variance = torch.clamp(variance, min=0.0)

            # Compute sigma (standard deviation)
            sigma = torch.sqrt(variance)              # Shape: [1, seq_len]

            # Extract token log probabilities and token IDs
            generated_token_ids = generated_ids.squeeze(0)
            token_log_probs = log_probs[0, torch.arange(generated_ids.shape[1]), generated_token_ids]

            # Decode the generated tokens to get the response
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Save
            data[i]['mu'] = mu.squeeze(0).cpu().tolist()
            data[i]['sigma'] = sigma.squeeze(0).cpu().tolist()
            data[i]['token_log_probs'] = token_log_probs.cpu().tolist()
            data[i]['token_ids'] = generated_token_ids.cpu().tolist()
            data[i]['generated_code'] = response

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
