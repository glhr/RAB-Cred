import glob


results_test_set_folder = "experiment_results/experiment_results_test200"
results_val_set_folder = "experiment_results/experiment_results_val"

test_set_csv_metadata = "../RAB-Cred/test_set200_metadata.csv"
test_set_csv_labels = "../RAB-Cred/test_set200_labels.csv"
val_set_csv_metadata = "../RAB-Cred/val_set_metadata.csv"
val_set_csv_labels = "../RAB-Cred/val_set_labels.csv"

# get experiments from experiment_results folder
experiment_folders_test = glob.glob(f"../{results_test_set_folder}/*")
experiment_folders_val = glob.glob(f"../{results_val_set_folder}/*")

num_params_per_model = {
    "bigscience/bloomz-7b1": 7.0,
    "microsoft/phi-4": 14.0,
    "deepcogito/cogito-v1-preview-qwen-14B": 14.0,
    "ibm-granite/granite-4.0-micro": 3.0,
    "openai/gpt-oss-20b": 20.0,
    "CohereLabs/aya-expanse-8b": 8.0,
    "microsoft/Phi-4-reasoning": 14.0,
    "microsoft/Phi-4-mini-instruct": 4.0,
    "ibm-granite/granite-4.0-h-tiny": 7.0,
    "CohereLabs/aya-expanse-32b": 32.0,
    "CohereLabs/c4ai-command-r-v01": 35.0
}

# these are the selected models for the test set evaluation
combos_of_interest = {
    'microsoft/phi-4': [('SP4', 'UP2'), ('SP3', 'UP2'), ('SP4', 'UP4')],
    'google/gemma-3-27b-it': [('SP4', 'UP1'), ('SP3', 'UP1'), ('SP4', 'UP1-FS')],
    'mistralai/Ministral-3-14B-Instruct-2512': [('SP4', 'UP3'), ('SP3', 'UP2'), ('SP5', 'UP2')],
    'mistralai/Mistral-Small-3.2-24B-Instruct-2506': [('SP3', 'UP1-FS'), ('SP5', 'UP2'), ('SP4', 'UP1')],
    'Qwen/Qwen3-30B-A3B-Instruct-2507': [('SP2', 'UP1-FS'), ('SP5', 'UP2'), ('SP5', 'UP3')]
}
