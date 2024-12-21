# Script used for tokenizing datasets for LoRA training
# Goal: Decrease training time by tokenizing datasets before training ensuring that the tokenization process is not repeated during training

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
TOKENIZED_DIR = "LoRA/Working/tokenized_datasets"
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

# Set padding side
tokenizer.padding_side = "right"

# Function to load processed datasets
def load_processed_datasets(isMath: bool):
    if os.path.exists(f"{PROCESSED_DIR}/linguistic") and os.path.exists(f"{PROCESSED_DIR}/meta_math"):
        print("Loading processed datasets from local disk...")
        if isMath:
            return DatasetDict.load_from_disk(f"{PROCESSED_DIR}/meta_math")
        else:
            return DatasetDict.load_from_disk(f"{PROCESSED_DIR}/linguistic")
    else:
        raise FileNotFoundError("Processed datasets not found in the specified directory.")

# Load dataset
processed_data = load_processed_datasets(isMath)
print(f"Loaded dataset: {processed_data}")

# Function to tokenize a single example
def tokenize(prompt, isMath):
    if isMath:
        # Concatenate query and response for math dataset
        input_text = prompt["query"] + "\n -->" + prompt["response"]

    else:
        # Concatenate query and response for linguistic dataset
        input_text = prompt["paragraph_generation_prompt"] + " \n -->" + prompt["claude_summary"]


    # Tokenize the concatenated input
    inputs = tokenizer(input_text, max_length=1024, truncation=True, padding="max_length")
    
    return {
        "input_ids": inputs["input_ids"], # The tokenized query and response
        "attention_mask": inputs["attention_mask"], # Attention mask for the input
        "labels": inputs["input_ids"], # The labels (response) for training
    }

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
