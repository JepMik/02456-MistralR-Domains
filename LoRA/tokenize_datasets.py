# Script to tokenize datasets and save them locally for reuse

# Imports
import torch
from transformers import AutoTokenizer
from datasets import DatasetDict, load_from_disk
import os
import sys

# Constants
MODELPATH = "ModelMistral"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
PROCESSED_DIR = "Baseline/processed_datasets"
TOKENIZED_DIR = "LoRA/tokenized_datasets"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Command-line argument to distinguish between math and linguistic datasets
arg = sys.argv[1]
isMath = arg == "Math"

# Load tokenizer
if os.path.exists(MODELPATH) and os.listdir(MODELPATH):
    print(f"Loading tokenizer from {MODELPATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODELPATH)
else:
    print(f"Downloading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"Saving tokenizer to {MODELPATH}...")
    tokenizer.save_pretrained(MODELPATH)

# Function to load processed datasets
def load_processed_datasets(isMath: bool):
    if os.path.exists(f"{PROCESSED_DIR}/linguistic") and os.path.exists(f"{PROCESSED_DIR}/meta_math"):
        print("Loading processed datasets from local disk...")
        if isMath:
            return DatasetDict.load_from_disk(f"{PROCESSED_DIR}/meta_math_tokenized")
        else:
            return DatasetDict.load_from_disk(f"{PROCESSED_DIR}/linguistic_tokenized")
    else:
        raise FileNotFoundError("Processed datasets not found in the specified directory.")

# Load dataset
processed_data = load_processed_datasets(isMath)
print(f"Loaded dataset: {processed_data}")

# Function to tokenize a single example
def tokenize(prompt, isMath):
    if isMath:
        inputs = tokenizer(
            prompt["query"], max_length=512, truncation=True, padding="max_length"
        )
        outputs = tokenizer(
            prompt["response"], max_length=512, truncation=True, padding="max_length"
        )
    else:
        inputs = tokenizer(
            prompt["paragraph_generation_prompt"], max_length=512, truncation=True, padding="max_length"
        )
        outputs = tokenizer(
            prompt["claude_summary"], max_length=512, truncation=True, padding="max_length"
        )
    return {
        "input_ids": inputs["input_ids"], # input_ids: The token indices in the vocabulary
        "attention_mask": inputs["attention_mask"], # attention_mask: The attention mask that indicates the valid values for input
        "labels": outputs["input_ids"], # labels: The token indices in the vocabulary
    } # Used for computing the loss during training

# Function to tokenize datasets
def tokenize_data(dataDict, isMath):
    print("Tokenizing dataset...")
    for key in dataDict:
        dataDict[key] = dataDict[key].map(lambda x: tokenize(x, isMath))
    return dataDict

# Tokenize data
tokenizer.pad_token = tokenizer.eos_token
tokenized_data = tokenize_data(processed_data, isMath)


# Save tokenized data
save_path = f"{TOKENIZED_DIR}/meta_math_tokenized" if isMath else f"{TOKENIZED_DIR}/linguistic_tokenized"
os.makedirs(TOKENIZED_DIR, exist_ok=True)
tokenized_data.save_to_disk(save_path) 
print(f"Tokenized dataset saved to {save_path}!")