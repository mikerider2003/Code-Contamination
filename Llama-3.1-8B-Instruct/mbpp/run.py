import os
import json
from datasets import load_dataset
import torch
import transformers
import torch.nn.functional as F
from tqdm import tqdm
import argparse

def convert_huggingface_data_to_list_dic(dataset, split):
    dataset = dataset[split]
    all_data = []
    for task in dataset:
        all_data.append(task)

    enhance_description(all_data)

    return all_data

# Task enhancment to make the code fit the tests
def enhance_description(data):
    for t_i in range(len(data)):
        # Add testcasses for better accuracy
        test_list = ""
        for test in data[t_i]['test_list']:
            test_list += test + "\n"

        # Enhancment + Description + Function
        data[t_i]['full_description'] = f"{data[t_i]['prompt']}\n\nIt must pass following tests:\n{test_list}"
        # print(data[t_i]['full_description'])
        # print("\n------------------\n")

def generate_code(data, output_filename="output.json"):
    # Load the model
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    with tqdm(total=len(data), desc="Generating Code") as pbar:
        for i in range(len(data)):
            prompt = data[i]['full_description']

            messages = [
                {"role": "system", "content": "You will get Python programming problem. You have to reply only with executable python code, no extra text."},
                {"role": "user", "content": prompt},
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=256,
            )
            response = outputs[0]["generated_text"][-1]["content"]
            
            # Cannot find how to get logits out of this
            raise NotImplementedError
            
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
    parser = argparse.ArgumentParser()
    parser.add_argument("split", help="Choose to use 'train' or 'test' split", type=str, choices=['train', 'test'])
    
    args = parser.parse_args()

    # Load dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized")
    data_converted = convert_huggingface_data_to_list_dic(dataset=ds, split=args.split)

    # Generate Code
    generate_code(data=data_converted, output_filename=f"mbpp_{args.split}.json")
