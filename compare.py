import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import concurrent.futures
import time


def load_saved_data(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"The file {filename} does not exist.")
        return []

def apply_on_dataset(data):
    with tqdm(total=len(data), desc="Processing dataset tasks", leave=False) as task_bar:
        for task in data:        
            # Clear the code
            cleared_code = clear_unwanted_code(task["generated_code"])

            # For OpenAI HumanEval there is need to generate test_list
            if "test_list" not in task:
                task["test_list"] = create_test_list(task["test"], task["generated_code"])
            
            # Test the code
            test_result = run_tests(cleared_code, task["test_list"])

            # Find mink++ Scores
            if "mkpp" not in task:
                mkpp = apply_mink_plus_plus(task)
                task["mkpp"] = mkpp

            # Save
            task["cleared_code"] = cleared_code
            task["test_result"] = test_result

            task_bar.update(1)
            
    return data

def clear_unwanted_code(response):
    if "python" in response and "```" in response:
        return response.split("python")[1].split("```")[0]
    return response

def create_test_list(test, code):
    f_name = None
    if "def" in code and "(" in code:
        f_name = code.split("def")[1].split("(")[0]

    parts = test.split("assert")

    test_list = []
    for part in parts[1:]:
        if f_name != None:
            part = part.replace("candidate", f_name)
        cleaned = part.replace("\n", "")
        test_list.append("assert" + cleaned)

    return test_list

def execute_with_timeout(code_snippet, namespace, timeout=30):
    """
    Executes a code snippet (using exec) within a specified timeout.
    """
    def exec_code():
        exec(code_snippet, namespace)
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(exec_code)
            future.result(timeout=timeout)  # Wait for the execution to finish or timeout
        return True  # Execution succeeded
    except concurrent.futures.TimeoutError:
        print(f"Execution timed out after {timeout} seconds.")
    # except Exception as e:
    #     print(f"Execution failed with error: {e}")
    return False  # Execution failed or timed out


def run_tests(code, test_list):
    namespace = {}
    passed_tests = 0
    total_tests = len(test_list)

    try:
        exec(code, namespace)
        for test in test_list:
            try:
                exec(test, namespace)
                passed_tests += 1

                # Alrenative because mbpp_test[127] stuck whole code in infinite loop
                # if execute_with_timeout(test, namespace, timeout=30):
                #     passed_tests += 1
            except Exception:
                pass 
    except Exception:
        pass 

    test_ratio = calculate_test_ratio(passed_tests, total_tests)

    return passed_tests, total_tests, test_ratio

def calculate_test_ratio(passed_tests, total_tests):
    if total_tests > 0:
        test_ratio = passed_tests / total_tests
    else:
        test_ratio = 0.0
    return test_ratio

def apply_mink_plus_plus(task, ratio_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    """
    Computes mink++ scores for the given task data and ratios.

    Parameters:
    - task: Dictionary containing 'mu', 'sigma', 'token_log_probs', 'token_ids'.
    - ratio_values: List of ratios to compute mink++ scores for.

    Returns:
    - scores: Dictionary mapping ratio labels to mink++ mean scores.
    """
    # Extract data from task
    mu = np.array(task['mu'])                  # Shape: [seq_len]
    sigma = np.array(task['sigma'])            # Shape: [seq_len]
    token_log_probs = np.array(task['token_log_probs'])  # Shape: [seq_len]
    
    # Ensure mu, sigma, and token_log_probs have the same length
    seq_len = len(token_log_probs)
    assert len(mu) == seq_len, "Length of mu does not match sequence length."
    assert len(sigma) == seq_len, "Length of sigma does not match sequence length."
    
    # Handle cases where sigma is zero or near zero to avoid division by zero
    sigma = np.where(sigma == 0, 1e-8, sigma)
    
    # Compute mink++ scores
    mink_plus = (token_log_probs - mu) / sigma  # Shape: [seq_len]
    
    # Initialize dictionary to store scores
    scores = {}
    
    # For each ratio, compute the mean of the lowest mink++ scores
    for ratio in ratio_values:
        # Determine the number of tokens to consider
        k_length = max(1, int(seq_len * ratio))
        
        # Sort the mink++ scores in ascending order
        sorted_scores = np.sort(mink_plus)
        
        # Select the lowest k_length scores
        lowest_scores = sorted_scores[:k_length]
        
        # Compute the mean of these lowest scores
        mean_score = np.mean(lowest_scores)
        
        # Store the result in the dictionary
        scores[f'mink++_{ratio}'] = mean_score
    
    return scores


def plot_distributions_two_datasets(data_1, data_2, title="", legend_labels=("label1", "label2")):
    mkpp_d1 = []
    mkpp_d2 = []

    for task in data_1:
        if "mkpp" in task:
            mkpp_d1.append(task["mkpp"])
    for task in data_2:
        if "mkpp" in task:
            mkpp_d2.append(task["mkpp"])

    df_1 = pd.DataFrame(mkpp_d1)
    df_2 = pd.DataFrame(mkpp_d2)

    df_2 = df_2.drop(index=19)  # To remove ouliar at row 19 OAIHE

    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    fig.tight_layout(pad=5)

    columns = df_1.columns
    bins = 30

    for i, ax in enumerate(axs.flat):
        if i < len(columns):
            col = columns[i]

            sns.histplot(df_1[col], bins=bins, kde=True, stat='density', color='orange', label=legend_labels[0], ax=ax, alpha=0.5)
            sns.histplot(df_2[col], bins=bins, kde=True, stat='density', color='blue', label=legend_labels[1], ax=ax, alpha=0.5)

            # sns.kdeplot(df_1[col], ax=ax, color='red', label='mbpp', alpha=0.7, linewidth=2)
            # sns.kdeplot(df_2[col], ax=ax, color='blue', label='OpenAI HumanEval', alpha=0.7, linewidth=2)
            # sns.rugplot(df_1[col], ax=ax, color='red', alpha=0.5)
            # sns.rugplot(df_2[col], ax=ax, color='blue', alpha=0.5)

            ax.set_title(f"Distribution of {col}")
            ax.legend()

    plt.suptitle(f"Distribution of two different datasets | {title}")
    plt.savefig(f"{title} distribution_plot.pdf")

    return 1

def success_rate(data):
    test_result = []

    for task in data:
        test_result.append(task["test_result"])
    
    df = pd.DataFrame(test_result)
    filtered_df_p = df[df.iloc[:, 2] == 1]
    
    full_working_code = len(filtered_df_p) / len(df)
    
    # filtered_df_f = df[df.iloc[:, 2] != 1]
    # average_pass_rate_when_fail = filtered_df_f.iloc[:, 2].mean()
    # print(average_pass_rate_when_fail)

    return full_working_code

def plot_d_pass_vs_fail(data, title=""):
    df = []

    for task in data:
        mink = []
        mink.append(task["test_result"][2])
        for mkpp in task['mkpp'].values():
            mink.append(mkpp)
        
        df.append(mink)
    
    column_names = ["test_result"] + list(task["mkpp"].keys())
    
    df = pd.DataFrame(df, columns=column_names)

    pass_df = df[df.iloc[:, 0] == 1].drop(df.columns[0], axis=1)
    fail_df = df[df.iloc[:, 0] != 1].drop(df.columns[0], axis=1)

    # print(pass_df)
    # print(fail_df)

    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    fig.tight_layout(pad=5)

    columns = pass_df.columns
    bins = 30

    for i, ax in enumerate(axs.flat):
        if i < len(columns):
            col = columns[i]

            sns.histplot(pass_df[col], bins=bins, kde=True, stat='density', color='green', label='Pass', ax=ax, alpha=0.5)
            sns.histplot(fail_df[col], bins=bins, kde=True, stat='density', color='red', label='Fail', ax=ax, alpha=0.5)

            ax.set_title(f"Distribution of {col}")
            ax.legend()
    
    plt.suptitle(f"{title} | Pass vs Fail")
    plt.savefig(f"{title} Pass_vs_Fail_plt.pdf")
    return 1

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify the data folder location.")
    parser.add_argument(
        "-d", "--data-folder",
        type=str,
        default=os.getcwd(), 
        help="Path to the data folder (default: current directory)"
    )
    args = parser.parse_args()

    data_folder = args.data_folder

    if not os.path.isdir(data_folder):
        print(f"Warning: {data_folder} is not a valid directory. Using the current directory instead.")
        data_folder = os.getcwd()

    print(f"Using data folder: {data_folder}")
    print(f"Files in the folder: {os.listdir(data_folder)}")

    try:
        # QWEN
        # MBPP Train
        data_1 = load_saved_data(os.path.join(data_folder, "qwen_mbpp_train.json"))
        
        # MBPP Test
        data_2 = load_saved_data(os.path.join(data_folder, "qwen_mbpp_test.json"))
        
        # WARNING !!! - Had to remove 127 generated code as it stucks whole pipeline
        removed_element = data_2.pop(127)
        print(removed_element)
        
        # PYTHIA
        # MBPP Train
        data_3 = load_saved_data(os.path.join(data_folder, "pythia_mbpp_train.json"))
        
        # MBPP Test
        data_4 = load_saved_data(os.path.join(data_folder, "pythia_mbpp_test.json"))
        
        print("Data successfully loaded!")
        
    except Exception as e:
        print(f"An error occurred while loading data: {e}")

    if data_1 and data_2 and data_3:
        with tqdm(total=4, desc="Clear, Test, Calc Mink on Dataset") as pbar:
            # Clear up Code + Testing + Mink_plus_plus
            data_1 = apply_on_dataset(data_1)
            pbar.update(1)
            
            data_2 = apply_on_dataset(data_2)
            pbar.update(1)
            
            data_3 = apply_on_dataset(data_3)
            pbar.update(1)

            data_4 = apply_on_dataset(data_4)
            pbar.update(1)

        # QWEN
        # MBPP | Train | Pass vs Fail 
        plot_d_pass_vs_fail(data_1, title="qwen_01_MBPP_train")

        # MBPP | Test | Pass vs Fail 
        plot_d_pass_vs_fail(data_2, title="qwen_01_MBPP_test")

        # MBPP | Train vs Test
        plot_distributions_two_datasets(data_1, data_2, title="qwen_02_MBPP_train vs Mbpp_test", legend_labels=("Mbpp_train", "Mbpp_test"))
        
        

        # Pythia
        # MBPP | Train | Pass vs Fail 
        plot_d_pass_vs_fail(data_3, title="pythia_01_MBPP_train")

        # MBPP | Test | Pass vs Fail 
        plot_d_pass_vs_fail(data_4, title="pythia_01_MBPP_test")

        # MBPP | Train vs Test
        plot_distributions_two_datasets(data_3, data_4, title="pythia_02_MBPP_train vs Mbpp_test", legend_labels=("Mbpp_train", "Mbpp_test"))



        # PYTHIA train vs QWEN train
        plot_distributions_two_datasets(data_1, data_3, title="03_PYTHIA train vs QWEN train", legend_labels=("QWEN", "PYTHIA"))

        # PYTHIA test vs QWEN test
        plot_distributions_two_datasets(data_2, data_4, title="03_PYTHIA test vs QWEN test", legend_labels=("QWEN", "PYTHIA"))
        
        print(f"QWEN_train: {success_rate(data_1):.2%}\nQwen_test: {success_rate(data_2):.2%}\nPythia_train: {success_rate(data_3):.2%}\nPythia_test: {success_rate(data_3):.2%}")




