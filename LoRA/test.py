## Load tokenized data
from datasets import load_from_disk

TOKENIZED_DIR = "LoRA/new/tokenized_datasets/meta_math_tokenized"

# Load tokenized data
dataset = load_from_disk(TOKENIZED_DIR)
#print(f"Loaded dataset: {dataset}")
#print(f"Example entry: {dataset['train'][0]}")

tokenized_data = load_from_disk(TOKENIZED_DIR)

#print(f"Loaded tokenized data: {tokenized_data}")
#print(f"Example entry: {tokenized_data['test'][0]}")

print(dataset["test"][0])