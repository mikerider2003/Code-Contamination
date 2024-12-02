import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_saved_data(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"The file {filename} does not exist.")
        return []

def apply_on_dataset(data):
    for task in data:
        
        # Clear the code
        cleared_code = clear_unwanted_code(task["generated_code"])

        # For OpenAI HumanEval there is need to generate test_list
        if "test_list" not in task:
            task["test_list"] = create_test_list(task["test"], task["generated_code"])
        
        # Test the code
        test_result = run_tests(cleared_code, task["test_list"])

        # Find mink++ Scores
        mkpp = apply_mink_plus_plus(task)

        # Save
        task["cleared_code"] = cleared_code
        task["test_result"] = test_result
        task["mkpp"] = mkpp
            
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

def stats(data):
    pass_ratio = []
    mkpp = []

    # Extract info
    for task in data:
        # Extraxt mbpp vals
        mkpp.append(task["mkpp"])

        # Extract passing ratios
        pass_ratio.append(task["test_result"][2])

    mkpp_df = pd.DataFrame(mkpp)
    print(mkpp_df.describe())

    pass_ratio_df = pd.DataFrame(pass_ratio)
    pass_ratio_df.columns = ['Pass Ratio']
    print(pass_ratio_df.describe())
    
    generate_boxplot(mkpp_df, pass_ratio_df.mean().iloc[0])

    return 1

def generate_boxplot(df, test_pass):
    # df = df.drop(index=19) # To remove ouliar at row 19 OAIHE

    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 6))

    sns.boxplot(data=df)

    plt.xticks(rotation=45)

    plt.xlabel('Threshold Levels')
    plt.ylabel('mink++ Scores')
    plt.title('Distribution of mink++ Scores Across Threshold Levels')

    plt.text(
    0.94,  # X position as a fraction of the figure width
    0.25,  # Y position as a fraction of the figure height
    f"{test_pass:.2%}",  # The text to display
    fontsize=72,  # Font size
    ha='right',  # Horizontal alignment
    transform=plt.gcf().transFigure  # Transform to figure coordinates
    )

    plt.tight_layout()
    plt.savefig('boxplot.pdf')

    return


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load saved data from a JSON file.")
    parser.add_argument('-f', '--filename', type=str, default='output.json', help='The name of the file to load data from.')
    
    args = parser.parse_args()
    
    data = load_saved_data(args.filename)

    if data:
        # Clear up Code + Testing + Mink_plus_plus
        data = apply_on_dataset(data)

        # Stats
        stats(data)

